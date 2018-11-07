'''
Library for spatially adaptive spectral mixture analysis (SASMA) as described
by Deng and Wu (2013) and Wu et al. (2014):

Deng, C., and C. Wu. 2013. A spatially adaptive spectral mixture analysis for mapping subpixel urban impervious surface distribution. Remote Sensing of Environment 133:62–70.

Wu, C., C. Deng, and X. Jia. 2014. Spatially constrained multiple endmember spectral mixture analysis for quantifying subpixel urban impervious surfaces. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 7 (6):1976–1984.

A typical SASMA analysis is performed within a search window:

1. Pick candidate endmembers using CART: fit_endmembers_class_tree()
2. Set the NoData value in the endmember raster array to zero, such that any
   NoData areas do not contribute to a weighted sum.
3. Calculate the weight of each endmember in the search window, typically
   using inverse distance weighting (IDW).
'''

import numpy as np
from sklearn import tree

class CARTLearner(object):
    def __init__(self, y_raster, *x_rasters, nodata=-9999):
        shp = y_raster.shape[1:]
        self.nodata = nodata
        self.y_raster = y_raster
        self.x_rasters = x_rasters
        self.n_features = len(x_rasters)
        self.x_features_array = np.concatenate(x_rasters, axis = 0)\
            .reshape((len(x_rasters), shp[0] * shp[1]))

    def fit_tree(self, params={}):
        '''
        Fits a (single) decision tree to the target endmembers, enabling learning
        of endmember type for all candidate endmember pixels. Arguments:
            y_raster    The raster with labeled classes; should be integer dtype
                        with a unique integer for every unique class label
            x_rasters   One or more rasters that contain predictive features
            nodata      The NoData value to ignore
            params      Additional parameters to pass when creating the
                        sklearn.tree.DecisionTreeClassifier instance
        Returns a tuple of:
            (tree.DecisionTreeClassifier instance, X, Y)
        '''
        assert all([r.shape[1:] == self.y_raster.shape[1:] for r in self.x_rasters]), 'x_raster arrays must all have the same shape as y_raster'
        shp = self.y_raster.shape[1:]

        x_masks = [np.where(x == self.nodata, 0, 1) for x in self.x_rasters]
        # Start with the y_raster so that its Nodata areas are shared
        x_combined_mask = np.where(self.y_raster == self.nodata, 0, 1)
        for mask in x_masks:
            x_combined_mask = np.multiply(x_combined_mask, mask)

        # Repeat the mask for every input feature
        x_combined_mask = np.repeat(x_combined_mask, self.n_features, axis = 0)\
            .reshape((self.n_features, shp[0] * shp[1]))

        # Apply the same NoData mask to all X rasters and the Y raster and combine
        #   all prediction features (as rasters) into a single raster array
        x_raster_masked = np.where(x_combined_mask == 0, self.nodata, self.x_features_array)
        y_raster_masked = np.where(x_combined_mask[0,...] == 0, self.nodata,
            self.y_raster.reshape((1, shp[0] * shp[1])))[0]

        # Create conformable label and prediction feature arrays
        x_array = x_raster_masked[:,x_raster_masked[0,:] != self.nodata]
        y_array = y_raster_masked[y_raster_masked != self.nodata]
        assert x_array.shape[-1] == y_array.shape[0], 'Labels and prediction features are not aligned (a problem filtering NoData?)'

        self.classifier = tree.DecisionTreeClassifier(**params)
        self.last_fit = self.classifier.fit(x_array.T, y_array)
        return (self.last_fit, x_array.T, y_array)

    def predict(self, fit=None, features=None):
        '''
        Predict the class labels (e.g., endmember types) based on an existing
        tree fit and new predictive features. Arguments:
            fit         The result of tree.DecisionTreeClassifier.fit(); uses
                        the last fit model if None.
            features    The new X array/ new predictive features to use;
                        should be (p x n), n samples with p features.
        '''
        if fit is None: fit = self.last_fit
        if features is None: features = self.x_features_array
        prediction = fit.predict(features.T)
        return prediction.reshape(self.y_raster.shape)


def eye(size, band_num=None):
    '''
    Generates an eye-shaped (or "donut-shaped") binary kernel for filtering/
    moving windows, e.g., the 3-by-3 case:
        1 1 1
        1 0 1
        1 1 1
    Arguments:
        size        The size of the window (sides of equal length)
        band_num    The number of bands to output; resulting raster array
                    will be (p x n x n) where n is the size and p is the
                    band_num
    '''
    # Get the index of the center, on either axis
    c = int(np.floor(np.median(range(0, size))))
    eye_win = np.ones((size, size))
    eye_win[c,c] = 0

    if band_num is not None:
        eye_win = np.repeat(eye_win.reshape((1, size, size)),
            band_num, axis = 0)

    return eye_win


def kernel_idw_l1(size, band_num=None, normalize=False, moore_contiguity=False):
    '''
    Generates an inverse distance weighting (IDW) map simply by assigning
    each cell of a uniform grid an equal weight based on the L1 norm or
    Manhattan distance. If Moore contiguity is used, i.e., the cells on
    the diagonal are assigned a distance of 1 unit, then the result is
    slightly different from the L1 norm:
    With Moore contiguity:      With Von Neumann contiguity:
        2 2 2 2 2                   4 3 2 3 4
        2 1 1 1 2                   3 2 1 2 3
        2 1 0 1 2                   2 1 0 1 2
        2 1 1 1 2                   3 2 1 2 3
        2 2 2 2 2                   4 3 2 3 4
    Arguments:
        size        The size of the window (sides of equal length)
        band_num    The number of bands to output; resulting raster array
                    will be (p x n x n) where n is the size and p is the
                    band_num
        normalize   If True, the weights in each band will sum to one
        moore_contiguity
                    Uses Moore's contiguity (or Queen's rule adjacency) in
                    weighting; if False, uses Von Neumann weighting scheme.
    '''
    # Get the index of the center, on either axis
    c = int(np.floor(np.median(range(0, size))))
    window = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            if i == j == c:
                window[i,j] = 0
                continue

            window[i,j] = 1 / sum((abs(j - c), abs(i - c)))

            if moore_contiguity:
                raise NotImplementedError("Moore contiguity/ Queen's rule not supported")

    if normalize:
        # Constraint: Sum of all weights must be equal to one IN EACH BAND
        window = window / np.sum(window)

    if band_num is not None:
        window = np.repeat(window.reshape((1, size, size)),
            band_num, axis = 0)

    return window
