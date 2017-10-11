'''
A collection of tools for visualizing data associated with LSMA, particularly
feature spaces. When run as a standalone script, can plot the feature space
of any raster (will not perform the MNF transform).
'''

import sys
import os
import re
import numpy as np
import pysptools.util as sp_utils
from unmixing.transform import mnf_rotation
from unmixing.lsma import point_to_pixel_geometry, ravel_and_filter
from unmixing.utils import as_array, binary_mask, pixel_to_geojson, pixel_to_xy, xy_to_pixel, spectra_at_xy, subarray
from osgeo import gdal, ogr, osr
from pylab import plt, figure
from matplotlib.path import Path as VectorPath
import matplotlib.patches as patches

KML_DOC_TEMPLATE ='<?xml version="1.0" encoding="utf-8" ?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>%s%s</Document></kml>'
KML_POLY_STYLE = '''<Style id="default">
<LineStyle><color>ff0000ff</color><width>3</width></LineStyle>
<PolyStyle><color>000000ff</color><colorMode>normal</colorMode></PolyStyle>
</Style>'''
KML_POLY_TEMPLATE = '''<Placemark><description>%s</description><styleUrl>#default</styleUrl>
<MultiGeometry>%s%s</MultiGeometry></Placemark>'''

class LSMAPlot(object):
    def __init__(self, path=None, mask=None, cut_dim=None, ravel=True, transform=True, nodata=None, feature_limit=90000, selected_feature_limit=30, epsg=None, keyword=None, verbose=False):
        self.__nodata__ = nodata
        self.__raveled__ = ravel
        self.__limit__ = feature_limit
        self.__sel_limit__ = selected_feature_limit
        self.__verbose__ = verbose
        self.epsg = epsg
        self.size = (9, 9)
        self.dpi = 72

        if path is not None:
            assert os.path.exists(path), 'No such file or directory'
            ds = gdal.Open(path) # (p, lat, lng)
            self.keyword = keyword # A "nickname" for this raster
            self.__wd__ = os.path.dirname(path)
            self.__gt__ = ds.GetGeoTransform()
            self.__wkt__ = ds.GetProjection()
            self.spatial_ref = {
                'gt': self.__gt__,
                'wkt': self.__wkt__
            }

            if keyword is None:
                # Look for a date (7-8 numbers) set off by underscores
                date_match = re.compile(r'.*_(?P<date>\d{7,8})_.*').match(os.path.basename(path))
                if date_match is not None:
                    self.keyword = date_match.groups()[0]

            # Apply the MNF transformation?
            if transform:
                self.features = mnf_rotation(ds.ReadAsArray()) # (lng, lat, p)

            else:
                self.features = ds.ReadAsArray().transpose()

            if cut_dim:
                # Get rid of extraneous dimensionality
                self.features = self.features[...,0:cut_dim]
                ds = None

            # Apply a mask?
            if mask is not None:
                if type(mask) == str:
                    mask, gt, wkt = as_array(mask)

                else:
                    if not isinstance(mask, np.ndarray):
                        mask = mask.ReadAsArray()

                self.features = binary_mask(self.features.transpose(), mask, nodata=nodata).transpose()
                mask = None

            # Create random features
            self.rfeatures = self.features.copy().reshape((
                self.features.shape[0] * self.features.shape[1],
                self.features.shape[2]))
            np.random.shuffle(self.rfeatures)

            # Limit the size of the stored array; keep first 90,000 (300*300)
            if ravel and nodata is not None:
                # Remove all "rows" (pixels) where there is a NoData value in any "column" (band)
                self.rfeatures = self.rfeatures[(self.rfeatures != nodata).any(axis=1),:]

                if self.__limit__ is not None:
                    self.rfeatures = self.rfeatures[0:self.__limit__,:]

            else:
                self.rfeatures = self.rfeatures.reshape(self.features.shape)
                # If a limit was specified, select the first N random pixels
                if self.__limit__ is not None:
                    r = int(np.sqrt(self.__limit__))
                    self.rfeatures = self.rfeatures.reshape(self.features.shape)[0:r,0:r,:]

            ds = None

    def __filter_spectra_at__(self, xy):
        '''
        Filters out the spectra at the given pixel coordinates from the MNF
        feature space.
        '''
        assert not self.__raveled__, 'Cannot do this when the input array is raveled'
        shp = self.features.shape
        ref = self.features.transpose()[xy[1], xy[0], :]
        arr = self.features.transpose().reshape(shp[-1],
            shp[0]*shp[1]).swapaxes(0, 1)

        idx = np.apply_along_axis(lambda a: not np.array_equal(a, ref), 1, arr)
        return arr[idx, :].swapaxes(0, 1)

    def __spectra__(self, points, dd, scale, domain, nodata=None):
        '''
        Accesses spectral profiles from the stored features for the given
        point coordinates. If `nodata` argument is provided, these are filtered
        from the spectra.
        '''
        assert not self.__raveled__, 'Cannot do this when the input array is raveled'
        spectra = spectra_at_xy(self.features.transpose(),
            points, dd=dd, **self.spatial_ref) * scale

        if nodata is not None:
            nodata_array = np.full((1, 6), nodata)
            spectra = spectra[np.all(spectra != nodata_array, 1), :]

        return spectra


class FeatureSpace(LSMAPlot):
    '''
    A plotting utility for displaying 2D slices of the feature space. The data
    are stored as an HSI cube; the input array is transposed.
    '''
    def __init__(self, *args, **kwargs):
        super(FeatureSpace, self).__init__(*args, **kwargs)
        self.__drawing_index__ = 0 # How many rectangles have been drawn?

    def on_draw(self, output_dir=None):
        def get_data_in_selection(c1, c2):
            condition = np.logical_and(c1, c2).reshape(shp2)
            # Get the X and Y pixel coordinates within the bounding boxes
            cx = ravel_and_filter(np.where(condition, xdata, nodata_array).T, nodata = self.__nodata__)
            cy = ravel_and_filter(np.where(condition, ydata, nodata_array).T, nodata = self.__nodata__)

            # Zip the X and Y arrays into an X,Y array
            # NOTE: We flip the Y and X here because the `xdata` are column
            #   indices and `ydata` are row indices, but as pixel coordinates
            #   the row number is a Y-axis cordinate, column number is
            #   an X-axis coordinate
            return np.dstack((cy[:,0], cx[:,0]))

        shp = self.features.shape
        shp2 = (shp[0], shp[1], 1) # Shape for a single band
        # Array of X coordinates
        xdata = np.repeat([np.arange(0, shp[1])], shp[0], axis = 0).reshape(shp2)
        # Array of Y coordinates
        ydata = np.repeat([np.arange(0, shp[0])], shp[1], axis = 0).T.reshape(shp2)
        nodata_array = np.ones(shp2) * self.__nodata__ # Make NoData array
        tmp = self.features.reshape((shp[0] * shp[1], shp[-1]))

        # Start with top-left, end with bottom-right
        if self.x0 < self.x1:
            if self.y0 > self.y1:
                selection = get_data_in_selection(
                np.logical_and(tmp[:,0] > self.x0, tmp[:,1] < self.y0),
                np.logical_and(tmp[:,0] < self.x1, tmp[:,1] > self.y1))

            # Start with bottom-left, end with top-right
            else:
                selection = get_data_in_selection(
                np.logical_and(tmp[:,0] > self.x0, tmp[:,1] > self.y0),
                np.logical_and(tmp[:,0] < self.x1, tmp[:,1] < self.y1))

        # Start with bottom-right, end with top-left
        else:
            if self.y0 < self.y1:
                selection = get_data_in_selection(
                np.logical_and(tmp[:,0] < self.x0, tmp[:,1] > self.y0),
                np.logical_and(tmp[:,0] > self.x1, tmp[:,1] < self.y1))

            # Start with top-right, end with bottom-left
            else:
                selection = get_data_in_selection(
                np.logical_and(tmp[:,0] < self.x0, tmp[:,1] < self.y0),
                np.logical_and(tmp[:,0] > self.x1, tmp[:,1] > self.y1))

        # Limit to N random features
        if selection.shape[1] >= self.__sel_limit__:
            rfeatures = np.random.choice(np.arange(0, selection.shape[1]),
                size = self.__sel_limit__, replace = False)
            rfeatures.sort() # Make it easier to get iterate through them in order
            selection = selection[:,rfeatures,:]

        file_path = os.path.join((output_dir or self.__wd__),
            'FeatureSpace_selection_%s_%d' % ((self.keyword or ''), self.__drawing_index__))
        points = pixel_to_xy(selection[0,:], self.__gt__, self.__wkt__, dd = False)
        points_dd = pixel_to_xy(selection[0,:], self.__gt__, self.__wkt__, dd = True)

        # If a source EPSG is known, convert the output to KML
        if self.epsg is not None:
            # Convert to WGS 84
            poly_geom = point_to_pixel_geometry(points, source_epsg = self.epsg, target_epsg = 4326)

            # We want to create Placemarks with both a Point and
            #   a Polygon geometry in each
            pmarks = []
            for i, poly in enumerate(poly_geom):
                pm = KML_POLY_TEMPLATE % (
                    ','.join(map(str, points[i])), # The (projected) coordinates
                    '<Point><coordinates>%f,%f</coordinates></Point>' % points_dd[i],
                    poly.ExportToKML() # The KML Polygon feature
                )
                pmarks.append(pm)
            doc = KML_DOC_TEMPLATE % (KML_POLY_STYLE, ''.join(pmarks))
            # Write out the coordinates as a KML file
            with open('%s.kml' % file_path, 'w') as stream:
                stream.write(doc)

            if self.__verbose__:
                sys.stdout.write("Wrote selection's coordinates in geographic space to: %s.kml\n" % file_path)

        else:
            sys.stdout.write("Warning: Source SRS not known; cannot output KML file\n")

    def on_reset(self):
        # A Rectangle for the interactive plot
        self.rect = patches.Rectangle((0,0), 1, 1, edgecolor=(1, 0, 0), facecolor=(1, 0, 0, 0.4))
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.__drawing_index__ += 1

    def on_press(self, event):
        if all((self.x0, self.y0)):
            self.x1 = event.xdata
            self.y1 = event.ydata

            # If both corners are defined
            if all((self.x0, self.y0, self.x1, self.y1)):
                self.rect.set_width(self.x1 - self.x0)
                self.rect.set_height(self.y1 - self.y0)
                self.rect.set_xy((self.x0, self.y0))
                self.ax.add_patch(self.rect)
                self.ax.figure.canvas.draw()
                self.on_draw()
                self.on_reset()

        else:
            self.x0 = event.xdata
            self.y0 = event.ydata

    def plot_eigenvalues(self):
        '''
        Plots the amount of variance explained by an increasing number of
        dimensions in the data.
        '''
        # Obtain the covariance matrix
        m, n, p = self.features.shape
        cov_m = sp_utils.cov(np.reshape(self.features, (m * n, p)))

        # Compute the eigenvalues
        eigenvals = np.linalg.eig(cov_m)[0]
        eigenvals_p = eigenvals / sum(eigenvals)
        plt.plot(eigenvals_p)

    def plot_feature_space(self, m=0, n=1, c=None, r=300, hold=False,
        xlim=None, ylim=None, xtpl='MNF %d', ytpl='MNF %d', alpha=0.5,
        stitle='MNF Feature Space: Axes %d and %d', interact=False):
        '''
        Create a 2D projection of the feature space and display it.
        '''
        self.__dims__ = (m, n, c) # Remember these indices

        # Create a new figure of size 9x9 points, using 72 dots per inch
        fig = figure(figsize=self.size, dpi=self.dpi)
        self.ax = fig.add_subplot(111)
        defaults = {
            'linewidths': (0,),
            's': (30,),
            'cmap': 'YlGnBu',
            'alpha': alpha
        }

        if self.__raveled__:
            if c is not None:
                self.ax.scatter(self.rfeatures[:,m], self.rfeatures[:,n],
                    c=self.rfeatures[:,c], **defaults)

            else:
                self.ax.scatter(self.rfeatures[:,m], self.rfeatures[:,n], **defaults)

        else:
            i = j = r # Select square subsets
            if c is not None:
                # Plot the data; if a third dimension in color is requested...
                self.ax.scatter(self.rfeatures[0:i,0:j,m], self.rfeatures[0:i,0:j,n],
                    c=self.rfeatures[0:i,0:j,c], **defaults)

            else:
                self.ax.scatter(self.rfeatures[0:i,0:j,m], self.rfeatures[0:i,0:j,n],
                    **defaults)

        if c is not None:
            plt.colorbar(orientation='vertical')
            t = '2D Projection with Axis %d in Color' % (c + 1)

        else:
            t = '2D Projection'

        # Allow users to change the x- and y-axis limits
        axes = plt.gca()
        if xlim is not None:
            axes.set_xlim(xlim)

        if ylim is not None:
            axes.set_ylim(ylim)

        plt.xlabel(xtpl % (m + 1), fontsize=14)
        plt.ylabel(ytpl % (n + 1), fontsize=14)
        plt.suptitle(stitle % (m + 1, n + 1),
            fontsize=18, fontweight='bold')
        plt.title(t, fontsize=16)

        if not hold:
            plt.show()

        if not hold and interact:
            self.on_reset()
            self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
            return self.ax

    def plot_2d_mixing_space(self, features, hold=False):
        '''
        Draws a 2D (triangular) mixing space.
        '''
        codes = [VectorPath.MOVETO, VectorPath.LINETO, VectorPath.LINETO, VectorPath.CLOSEPOLY]
        verts = features[...,0:2].tolist()
        verts.append((0, 0)) # Dummy vertex
        path = VectorPath(verts, codes)
        patch = patches.PathPatch(path, facecolor='black', alpha=0.3, lw=0)
        plt.gca().add_patch(patch)

        if not hold:
            plt.show()

    def plot_xy_points(self, coords, fmt='r+', labels=None, lpos=None, dd=True):
        '''
        Add point(s) to the plot, given by their longitude-latitude (XY)
        coordinates, which will be displayed at the appropriate feature space
        coordinates.
        '''
        assert not self.__raveled__, 'Cannot do this when the input array is raveled'
        m, n, c = self.__dims__
        pcoords = xy_to_pixel(coords, gt = self.__gt__, wkt = self.__wkt__, dd = dd)

        if labels is not None and lpos is None:
            lpos = [(10, -10)] * len(labels)

        # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
        for i, (x, y) in enumerate(pcoords):
            spec = self.features[x,y,:]
            plt.plot(spec[m], spec[n], fmt, ms = 20, mew = 2)

            if labels is not None:
                plt.annotate(labels[i], xy = (spec[m], spec[n]), xytext = lpos[i],
                    textcoords = 'offset points', ha = 'right', va = 'bottom',
                    fontsize = 14, arrowprops = dict(arrowstyle = '->',
                    connectionstyle = 'arc3,rad=0'))

    def plot_spectral_points(self, spectra, fmt='r+', labels=None, lpos=None):
        '''
        Plots spectral profiles in feature space.
        '''
        m, n, c = self.__dims__
        assert type(spectra) in (list, tuple, np.ndarray), 'Expected spectra to be a list, tuple, or numpy array'

        if labels is not None and lpos is None:
            lpos = [(30, -30)] * len(labels)

        for i, spec in enumerate(spectra):
            plt.plot(spec[m], spec[n], fmt, ms = 20, mew = 2)

            if labels is not None:
                xy = (spec[m], spec[n])
                plt.annotate(labels[i], xy = xy, xytext = lpos[i],
                    textcoords = 'offset points', ha = 'right', va = 'bottom',
                    fontsize = 14, arrowprops = dict(arrowstyle = '->',
                    connectionstyle = 'arc3,rad=0'))


    def plot_spectral_profile(self, points, dd=False, scale=1,
            domain=np.array([1,2,3,4,5,7]), labels=None, xlab=None, ylab=None,
            ylim=(0, None), lloc='upper left', **kwargs):
        '''
        Assumes the domain is the Landsat TM/ETM+ bands minus the thermal IR
        channel (bands 1-5 and 7). If the spectra are in Landsat surface
        reflectance units, they should be scaled by 0.01 to get reflectance
        as a percentage (by 0.0001 to get it as a proportion).
        '''
        assert not self.__raveled__, 'Cannot do this when the input array is raveled'
        spectra = self.__spectra__(points, dd, scale, domain, self.__nodata__)
        xs = range(1, domain.shape[0] + 1)

        # Truncate the spectra if necessary
        if len(xs) != spectra.shape[1]:
            spectra = spectra[:,0:len(xs)]

        # Plot as lines
        lines = plot(xs, spectra.transpose(), linewidth=2, **kwargs)

        # Set the x-axis tick labels (e.g., skip band 6)
        plt.xticks(xs, domain)
        plt.ylim(ylim)

        if xlab is not None:
            plt.xlabel(xlab)

        if ylab is not None:
            plt.ylabel(ylab)

        if labels is not None:
            plt.legend(lines, labels, loc=lloc, frameon=False)

        plt.show()

    def plot_tasseled_cap(self, m=0, n=1, c=None, r=1000, xlim=None, ylim=None):
        '''
        A special feature space plot for Tasseled Cap brightness, greenness,
        and wetness plots.
        '''
        if len(sys.argv) > 5:
            r = int(sys.argv[5])

        xtpl = 'Tasseled Cap Brightness (TC%s)'
        if m == 0 and n == 1:
            ytpl = 'Tasseled Cap Greenness (TC%s)'

        elif m == 0 and n == 2:
            ytpl = 'Tasseled Cap Wetness (TC%s)'

        elif m == 2 and n == 1:
            xtpl = 'Tasseled Cap Wetness (TC%s)'
            ytpl = 'Tasseled Cap Greenness (TC%s)'

        else:
            raise NotImplemented('This function is not prepared for that combination of axes in that order')

        self.plot_feature_space(m=m, n=n, c=c, r=r, xtpl=xtpl, ytpl=ytpl,
            stitle='Tasseled Cap Components %s and %s', xlim=xlim, ylim=ylim)


def cumulative_freq_plot(rast, band=0, mask=None, bins=100, xlim=None, nodata=-9999):
    '''
    Plots an empirical cumulative frequency curve for the input raster array
    in a given band.
    '''
    if mask is not None:
        arr = binary_mask(rast, mask)

    else:
        arr = rast.copy()

    if nodata is not None:
        arr = subarray(arr)

    values, base = np.histogram(arr, bins=bins)
    cumulative = np.cumsum(values) # Evaluate the cumulative distribution
    plt.plot(base[:-1], cumulative, c='blue') # Plot the cumulative function
    plt.set_title('Empirical Cumulative Distribution: Band %d' % band)

    if xlim is not None:
        axes = plt.gca()
        axes.set_xlim(xlim)

    plt.show()
    return arr


def histogram(arr, valid_range=(0, 1), bins=10, normed=False, cumulative=False,
        file_path='hist.png', title=None):
    '''
    Plots a histogram for an input array over a specified range.
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(arr, np.ndarray):
        arr = arr.ReadAsArray()

    plt.hist(arr.ravel(), range=valid_range, bins=bins, normed=normed,
        cumulative=cumulative)
    if title is not None:
        plt.title(title)

    plt.savefig(file_path)


if __name__ == '__main__':
    path = sys.argv[1]
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    c = int(sys.argv[4])
    r = 1000

    if len(sys.argv) > 5:
        r = int(sys.argv[5])

    vis = FeatureSpace(path, transform=False)
    vis.plot_feature_space(m=m, n=n, c=c, r=r)
