'''
Library for spatially adaptive spectral mixture analysis (SASMA) as described
by Deng and Wu (2013) and Wu et al. (2014):

Deng, C., and C. Wu. 2013. A spatially adaptive spectral mixture analysis for mapping subpixel urban impervious surface distribution. Remote Sensing of Environment 133:62–70.

Wu, C., C. Deng, and X. Jia. 2014. Spatially constrained multiple endmember spectral mixture analysis for quantifying subpixel urban impervious surfaces. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 7 (6):1976–1984.

A typical SASMA analysis is performed within a search window:

1. Pick candidate endmembers using CART: CARTLearner.fit_tree()
2. Set the NoData value in the endmember raster array to zero, such that any
   NoData areas do not contribute to a weighted sum.
3. Calculate the weight of each endmember in the search window, typically
   using inverse distance weighting (IDW): interpolate_endmember_map()
'''

import numpy as np
from itertools import product
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import generic_filter
from sklearn import tree
from unmixing.utils import binary_mask

class CARTLearner(object):
    def __init__(self, y_raster, *x_rasters, nodata=-9999):
        shp = y_raster.shape[1:]
        self.nodata = nodata
        self.y_raster = y_raster
        self.x_rasters = x_rasters
        self.n_labels = shp[0]
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

    def predict(self, fit=None, features=None, probabilities=False):
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
        if probabilities:
            shp = self.y_raster.shape
            return fit.predict(features.T).T.reshape((self.n_labels, shp[1], shp[2]))

        return fit.predict(features.T).reshape(self.y_raster.shape)


def concat_endmember_arrays(*em_rast_arrays):
    '''
    Concatenates multiple (p x m x n) raster arrays for each of q endmembers
    into (c x q x p) endmember array, for multiple-endmember approaches
    inclusing SASMA, where c = m*n and p is the number of spectral bands.
    Arguments:
        em_rast_arrays  Any number of input (p x m x n) raster arrays
    '''
    shp = em_rast_arrays[0].shape
    # Transform to (c x p) for each of q endmembers
    vectors = [e.reshape((shp[0], shp[1] * shp[2])).T for e in em_rast_arrays]
    # Next, transform to (c x p x 1), then swap axes 1 and 2, resulting in
    #    (c x 1 x p) arrays, that are concatenated on axis 1
    return np.concatenate([
        e.reshape((shp[1] * shp[2], shp[0], 1)).swapaxes(1, 2) for e in vectors
    ], axis = 1)


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


def kernel_idw_l1(
        size, band_num=None, normalize=False, moore_contiguity=False):
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
                    weighting; if False, uses Von Neumann (or Rook's rule).
    '''
    # Get the index of the center, on either axis
    c = int(np.floor(np.median(range(0, size))))
    window = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            if i == j == c:
                window[i,j] = 0
                continue

            if moore_contiguity:
                window[i,j] = 1 / max((abs(j - c), abs(i - c)))

            else:
                window[i,j] = 1 / sum((abs(j - c), abs(i - c)))

    if normalize:
        # Constraint: Sum of all weights must be equal to one IN EACH BAND
        window = window / np.sum(window)

    if band_num is not None:
        window = np.repeat(window.reshape((1, size, size)),
            band_num, axis = 0)

    return window


def interpolate_endmember_map(
        spectra, em_locations, window, q=3, n=2, labels=None, cval=0,
        nodata=-9999, multiprocess=False):
    '''
    Creates synthetic image endmember maps (arrays); these endmembers are
    spatially interpolated from known endmember candidates. The end goal
    is a map (array) with an endmember spectra at every pixel, for one or
    more endmembers. Arguments:
        spectra         The multi-band raster from which candidate endmember
                        spectra will be synthesized; a (p x m x n) array.
        em_locations    A single band, classified raster array denoting the
                        locations of endmember candidates; a (1 x m x n) array.
        window          A 2D NumPy array that serves as the moving window,
                        kernel, or filter used in SASMA.
        q               The number of endmember classes to synthesize.
        n               The dimensionality of the spectral subspace.
        labels          The values in em_locations that uniquely identify.
                        each endmember class; default is [1,2,...,q]
        cval            The constant value: Replaces NoData and used in areas
                        outside the window; see NOTE below.
        nodata          The NoData value.
        multiprocess    True to generate multiple processes, one per
                        endmember-band combination, i.e., q*n processes.

    Example of this routine on a tiny numeric array:
        ex = np.where(np.ndarray((1,5,5), dtype=np.int16) % 2 == 0, 0, 1)
        em_map = np.multiply(ex.repeat(ex, 3, 0),
            np.round(np.random.rand(3,5,5), 2))
        NODATA = 0 # Example of a NoData value to filter
        window = np.ravel(kernel_idw_l1(3, band_num=1))
        avg_map = generic_filter(em_map[0,...],
            lambda x: np.sum(np.multiply(x, window)) / np.sum(
                np.multiply(np.where(x == NODATA, 0, 1), window)),
            mode = 'constant', cval = 0, footprint = np.ones((3,3)))
        # If you want to place the original (raw) values into the averaged map
        np.where(em_map[0,...] > 0, em_map[0,...], avg_map)

    NOTE: For performance, NoData areas are filled with zero (0); this way,
    they do not contribute to the spatial sum (i.e., sum([0,...,0]) == 0) and
    they can also be removed from the sum of weights that is the divisor. In
    most cases, this shouldn't be an issue (minimum noise fraction components
    almost never equal zero); however, if areas that equal zero should be
    considered in the weighted sum, simply change cval.
    '''
    shp = spectra.shape
    if labels is None:
        # In absence of user-defined endmember class labels, use integers
        labels = range(1, (q + 1))

    assert len(labels) <= spectra.shape[0], 'The spectra array must have p bands for p endmember class labels'
    masked_spectra = [ # Extract the individual endmember "band" images
        binary_mask(spectra[j,...].reshape((1, shp[1], shp[2])),
            np.where(em_locations == i, 1, 0), nodata = nodata,
                invert = True) for i, j in list(product(labels, range(0, n)))
    ]

    if multiprocess:
        with ProcessPoolExecutor(max_workers = len(masked_spectra)) as executor:
            result = executor.map(
                partial(interpolate_endmember_spectra, window = window,
                    nodata = nodata),
                masked_spectra)

        result = list(result)

    else:
        result = []
        for em_map in masked_spectra:
            result.append(
                interpolate_endmember_spectra(em_map, window, cval, nodata))

    # "Let's get the band back together again"
    synth_ems = []
    for i in range(0, q): # Group bands by endmember type
        synth_ems.append(np.concatenate(result[(i*n):((i+1)*n)], axis = 0))

    return synth_ems


def interpolate_endmember_spectra(em_map, window, cval=0, nodata=-9999):
    '''
    Spatially interpolates a single-band image using the given window; not
    intended for direct use, rather, it is a module-level function for use
    in a ProcessPoolExecutor's context as part of interpolate_endmember_map().
    Arguments:
        em_map  A single-band raster array with most, but not all, pixels
                masked; these are interpolated from the values of the
                unmasked pixels.
        window  A square array representing a moving window.
        cval    The constant value to use outside of the em_map array;
                should be set to zero for proper interpolation of endmember
                spectra.
    '''
    shp = em_map.shape
    w = np.max(window.shape) # Assume square window; longest of any equal side
    window = np.ravel(window) # For performance, used raveled arrays
    em_avg_map = generic_filter(
        # Fill NoData with zero --> no contribution to spatial sum
        np.where(em_map[0,...] == nodata, cval, em_map[0,...]),
            # Multiply em_map in window by weights, then divide by
            #   the sum of weights in those non-zero areas
            lambda x: np.sum(np.multiply(x, window)) / np.sum(
                np.multiply(np.where(x == cval, 0, 1), window)),
            mode = 'constant', cval = cval, footprint = np.ones((w,w)))
    return em_avg_map.reshape((1, shp[1], shp[2]))


def mask_unstable_abundances(abundances, band=1, nodata=-9999, tol=0.9999):
    '''
    SASMA abundance estimates can be unstable; similar spectra will
    sometimes be unmixed very differently, where one mixture results in an
    abundance estimate of 100% for a single endmember that accounts for far
    less than 100% in another mixture (when the sum-to-one constraint is
    used). The approach used to correct this, here, is to mask out pixels
    in the given band which are equal to 1.0 exactly, which may not be a
    good policy in all cases (obviously, some endmembers are equal to 1.0
    exactly in some cases). Arguments:
        raster_array    The input abundance map to be cleaned
        band            The band that is erroneously estimated to be 100%
                        for some pixels
    '''
    abundances[:,abundances[band,...] > tol] = nodata
    return abundances
