'''
Utilities and convenience functions for manipulating spatial data and
spectral data cubes.
Contains:

* `as_array()`
* `as_mask()`
* `as_raster()`
* `array_to_raster()`
* `array_to_raster_clone()`
* `binary_mask()`
* `cfmask()`
* `clip_raster()`
* `clip_raster_file()`
* `combine_dicts()`
* `combine_masks()`
* `composite()`
* `density_slice()`
* `dump_raster()`
* `mae()`
* `mask_ledaps_qa()`
* `mask_saturation()`
* `pixel_to_geojson()`
* `pixel_to_xy()`
* `rmse()`
* `spectra_at_idx()`
* `spectra_at_xy()`
* `stack_hdf_as_array()`
* `stack_hdf_as_geotiff()`
* `xy_to_pixel()`

'''
import json
import os
import re
from osgeo import gdal, gdal_array, gdalnumeric, ogr, osr
from PIL import Image, ImageDraw
import numpy as np

def as_array(path):
    '''
    A convenience function for opening a raster as an array and accessing its
    spatial information; takes a single string argument.
    '''
    ds = gdal.Open(path)
    arr = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    ds = None
    return (arr, gt, wkt)


def as_mask(path, nodata=-9999):
    '''
    Converts all non-zero values in all bands to ones.
    '''
    rast, gt, wkt = as_array(path)

    # Create a baseline raster
    base = np.empty((1, rast.shape[-2], rast.shape[-1]))
    base.fill(False)

    # Case of multiband raster
    if rast.ndim == 3:
        # Update the mask for nonzero values in any band
        for i in range(rast.shape[0]):
            np.logical_or(base, (rast[i,...].ravel() > 0).reshape(rast[i,...].shape), out=base)

        # Repeat the value of one (1) across the bands
        np.place(rast, base.repeat(rast.shape[0], axis=0), (1,))

    elif rast.ndim == 2:
        # Create a single band (dim-3 array)
        rast = rast.reshape((1, rast.shape[-2], rast.shape[-1]))

        # Update the mask for nonzero values in any band
        np.logical_or(base, (rast.ravel() > 0).reshape(rast.shape), out=base)

        # Repeat the value of one (1) across the bands
        np.place(rast, base, (1,))

    else:
        raise ValueError('Number of array dimensions must be 2 or 3')

    # Replace the NoData values
    rast[rast == nodata] = 0

    return (rast, gt, wkt)


def as_raster(path):
    '''
    A convenience function for opening a raster and accessing its spatial
    information; takes a single string argument.
    '''
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    return (ds, gt, wkt)


def array_to_raster(a, gt, wkt, xoff=None, yoff=None):
    '''
    Creates a raster from a given array, with optional x- and y-offsets
    if the array was clipped. Arguments:
        a       A NumPy array
        gt      A GDAL GeoTransform tuple
        wkt     Well-Known Text projection
        xoff    The offset in the x-direction; should be provided when clipped
        yoff    The offset in the y-direction; should be provided when clipped
    '''
    try:
        rast = gdal_array.OpenNumPyArray(a)

    except AttributeError:
        # For backwards compatibility with older version of GDAL
        rast = gdal.Open(gdalnumeric.GetArrayFilename(a))

    kwargs = dict()
    if xoff is not None and yoff is not None:
        kwargs = dict(xoff=xoff, yoff=yoff)

    rast.SetGeoTransform(gt)
    rast.SetProjection(wkt)

    return rast


def array_to_raster_clone(a, proto, xoff=None, yoff=None):
    '''
    Creates a raster from a given array and a prototype raster dataset, with
    optional x- and y-offsets if the array was clipped. Arguments:
        a       A NumPy array
        proto   A prototype dataset
        xoff    The offset in the x-direction; should be provided when clipped
        yoff    The offset in the y-direction; should be provided when clipped
    '''
    rast = gdal_array.OpenNumPyArray(a)

    kwargs = dict()
    if xoff is not None and yoff is not None:
        kwargs = dict(xoff=xoff, yoff=yoff)

    # Copy the projection info and metadata from a prototype dataset
    if type(proto) == str:
        proto = gdal.Open(proto)

    gdalnumeric.CopyDatasetInfo(proto, rast, **kwargs)

    return rast


def binary_mask(rast, mask, nodata=-9999, invert=False):
    '''
    Applies an arbitrary, binary mask (data in [0,1]) where pixels with
    a value of 1 are pixels to be masked out. Arguments:
        rast    A gdal.Dataset or a NumPy array
        mask    A gdal.Dataset or a NumPy array
        nodata  The NoData value; defaults to -9999.
        invert  Invert the mask? (tranpose meaning of 0 and 1); defaults to False.
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        rastr = rast.ReadAsArray()

    else:
        rastr = rast.copy()

    if not isinstance(mask, np.ndarray):
        maskr = mask.ReadAsArray()

    else:
        maskr = mask.copy()

    # Transform into a "1-band" array and apply the mask
    if maskr.shape != rastr.shape:
        maskr = maskr.reshape((1, maskr.shape[0], maskr.shape[1]))\
            .repeat(rastr.shape[0], axis=0) # Copy the mask across the "bands"

    # Mask out areas that match the mask (==1)
    if invert:
        rastr[maskr < 1] = nodata

    else:
        rastr[maskr > 0] = nodata

    return rastr


def cfmask(rast, mask=None, mask_path=None, nodata=-9999):
    '''
    Applies the CFMask algorithm results to the image as a mask; masks out
    water, cloud, shadow, and snow (if any).

    More information on the QA bands can be found on page 14 of this document:
    http://landsat.usgs.gov/documents/cdr_sr_product_guide.pdf

    Arguments:
        rast        A gdal.Dataset or a NumPy array
        mask        A gdal.Dataset or a NumPy array
        mask_path   The path to an EOS HDF4 CFMask raster
        nodata      The NoData value; defaults to -9999.
    '''
    if (mask is None and mask_path is None) or (mask is not None and mask_path is not None):
        raise ValueError('Either `mask` or `mask_path` must be provided; they cannot both be None and only one should be specified')

    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        rastr = rast.ReadAsArray()

    else:
        rastr = rast.copy()

    if mask is not None:
        if not isinstance(mask, np.ndarray):
            maskr = mask.ReadAsArray()

        else:
            maskr = mask.copy()

    if mask_path is not None:
        mask_ds = gdal.Open('HDF4_EOS:EOS_GRID:"%s":Grid:cfmask' % mask_path)
        maskr = mask_ds.ReadAsArray()
        mask_ds = None

    # Transform into a "1-band" array and apply the mask
    maskr = maskr.reshape((1, maskr.shape[0], maskr.shape[1]))\
        .repeat(rastr.shape[0], axis=0) # Copy the mask across the "bands"

    # Mask out areas that match the mask
    # 1 = Water, 2 = Shadow, 3 = Snow, 4 = Cloud
    rastr[maskr > 0] = nodata
    return rastr


def clean_mask(rast):
    '''
    Clips the values in a mask to the interval [0, 1]; values larger than 1
    become 1 and values smaller than 0 become 0.
    Arguments:
        rast    An input gdal.Dataset or numpy.array instance
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        rastr = rast.ReadAsArray()

    else:
        rastr = rast.copy()

    return np.clip(rastr, a_min=0, a_max=1)


def clip_raster(rast, features_path, gt=None, nodata=-9999):
    '''
    Clips a raster (given as either a gdal.Dataset or as a numpy.array
    instance) to a polygon layer provided by a Shapefile (or other vector
    layer). If a numpy.array is given, a "GeoTransform" must be provided
    (via dataset.GetGeoTransform() in GDAL). Returns an array. Clip features
    must be a dissolved, single-part geometry (not multi-part). Modified from:

    http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    #clip-a-geotiff-with-shapefile

    Arguments:
        rast            A gdal.Dataset or a NumPy array
        features_path   The path to the clipping features
        gt              An optional GDAL GeoTransform to use instead
        nodata          The NoData value; defaults to -9999.
    '''
    def array_to_image(a):
        '''
        Converts a gdalnumeric array to a Python Imaging Library (PIL) Image.
        '''
        i = Image.fromstring('L',(a.shape[1], a.shape[0]),
            (a.astype('b')).tostring())
        return i

    def image_to_array(i):
        '''
        Converts a Python Imaging Library (PIL) array to a gdalnumeric image.
        '''
        a = gdalnumeric.fromstring(i.tobytes(), 'b')
        a.shape = i.im.size[1], i.im.size[0]
        return a

    def world_to_pixel(geo_matrix, x, y):
        '''
        Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
        the pixel location of a geospatial coordinate; from:
        http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#clip-a-geotiff-with-shapefile
        '''
        ulX = geo_matrix[0]
        ulY = geo_matrix[3]
        xDist = geo_matrix[1]
        yDist = geo_matrix[5]
        rtnX = geo_matrix[2]
        rtnY = geo_matrix[4]
        pixel = int((x - ulX) / xDist)
        line = int((ulY - y) / xDist)
        return (pixel, line)

    if not os.path.exists(features_path):
        raise ValueError('File path to cut features does not exist')

    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        gt = rast.GetGeoTransform()
        rast = rast.ReadAsArray()

    # Create an OGR layer from a boundary shapefile
    features = ogr.Open(features_path)
    if features.GetDriver().GetName() == 'ESRI Shapefile':
        lyr = features.GetLayer(os.path.split(os.path.splitext(features_path)[0])[1])

    else:
        lyr = features.GetLayer()

    # Get the first feature
    poly = lyr.GetNextFeature()

    # Convert the layer extent to image pixel coordinates
    minX, maxX, minY, maxY = lyr.GetExtent()
    ulX, ulY = world_to_pixel(gt, minX, maxY)
    lrX, lrY = world_to_pixel(gt, maxX, minY)

    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)

    # If the clipping features extend out-of-bounds and ABOVE the raster...
    if gt[3] < maxY:
        # In such a case... ulY ends up being negative--can't have that!
        iY = ulY
        ulY = 0

    # Multi-band image?
    try:
        clip = rast[:, ulY:lrY, ulX:lrX]

    except IndexError:
        clip = rast[ulY:lrY, ulX:lrX]

    # Create a new geomatrix for the image
    gt2 = list(gt)
    gt2[0] = minX
    gt2[3] = maxY

    # Map points to pixels for drawing the boundary on a blank 8-bit,
    #   black and white, mask image.
    points = []
    pixels = []
    geom = poly.GetGeometryRef()
    pts = geom.GetGeometryRef(0)

    for p in range(pts.GetPointCount()):
        points.append((pts.GetX(p), pts.GetY(p)))

    for p in points:
        pixels.append(world_to_pixel(gt2, p[0], p[1]))

    raster_poly = Image.new('L', (pxWidth, pxHeight), 1)
    rasterize = ImageDraw.Draw(raster_poly)
    rasterize.polygon(pixels, 0) # Fill with zeroes

    # If the clipping features extend out-of-bounds and ABOVE the raster...
    if gt[3] < maxY:
        # The clip features were "pushed down" to match the bounds of the
        #   raster; this step "pulls" them back up
        premask = image_to_array(raster_poly)
        # We slice out the piece of our clip features that are "off the map"
        mask = np.ndarray((premask.shape[-2] - abs(iY), premask.shape[-1]), premask.dtype)
        mask[:] = premask[abs(iY):, :]
        mask.resize(premask.shape) # Then fill in from the bottom

        # Most importantly, push the clipped piece down
        gt2[3] = maxY - (maxY - gt[3])

    else:
        mask = image_to_array(raster_poly)

    # Clip the image using the mask
    try:
        clip = gdalnumeric.choose(mask, (clip, nodata))

    # If the clipping features extend out-of-bounds and BELOW the raster...
    except ValueError:
        # We have to cut the clipping features to the raster!
        rshp = list(mask.shape)
        if mask.shape[-2] != clip.shape[-2]:
            rshp[0] = clip.shape[-2]

        if mask.shape[-1] != clip.shape[-1]:
            rshp[1] = clip.shape[-1]

        mask.resize(*rshp, refcheck=False)

        clip = gdalnumeric.choose(mask, (clip, nodata))

    return (clip, ulX, ulY, gt2)


def clip_raster_file(rast_path, shp_path, output_path=None, nodata=-9999):
    '''
    Clips a raster file as provided by a file path. This method is not careful
    about NoData values and can lead to unpredictable results. Online clipping
    with clip_raster() is preferred. Arguments:
        rast_path       The path to the raster dataset
        shp_path        The path to a Shapefile with clipping features
        output_path     The path for the output clipped dataset
        nodata          The NoData value; defaults to -9999.
    '''

    def open_array(array, prototype_ds=None, xoff=0, yoff=0):
        '''
        This is basically an overloaded version of the gdal_array.OpenArray
        passing in xoff, yoff explicitly so we can pass these params off to
        CopyDatasetInfo.
        '''
        ds = gdal.Open(gdalnumeric.GetArrayFilename(array))

        if ds is not None and prototype_ds is not None:
            if type(prototype_ds) == str:
                prototype_ds = gdal.Open(prototype_ds)

            if prototype_ds is not None:
                gdalnumeric.CopyDatasetInfo(prototype_ds, ds, xoff=xoff, yoff=yoff)

        return ds

    clip, xoff, yoff, gt = clip_raster(gdal.Open(rast_path), shp_path,
        nodata=nodata)
    # Determine what the output file path should be
    if output_path is None:
        new_name = ''.join(('.'.join(os.path.basename(rast_path).split('.')[:-1]),
            '_clip'))
        output_path = ''.join((os.path.join(os.path.dirname(rast_path),
            new_name), '.tiff'))
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(output_path, open_array(clip,
        prototype_ds=rast_path, xoff=xoff, yoff=yoff))


def combine_dicts(dict1, dict2):
    '''
    Combines two dictionaries that have lists as values. Arguments:
        dict1   A dictionary
        dict2   Another dictionary
    '''
    d = dict()
    keys = set(dict1.keys()).union(dict2.keys())

    for key in keys:
        # Create new lists for missing keys
        try:
            d[key] = dict1[key]
        except KeyError:
            try:
                d[key] = dict2[key]
            except KeyError:
                pass
        # Combine lists
        try:
            d[key].extend(dict2[key])
        except KeyError:
            pass

    return d


def combine_masks(*masks, multiply=False):
    '''
    All masks must have the same shape. When multiply=False, the combined
    mask takes on a value of 1 where any individual mask has 1; this
    produces a "greedy" mask. When multiply=True, the combined mask
    takes on a value of 0 where any individual mask has 0 (by multiplying
    individual values); this produces a "conservative" mask. If 1 is "bad"
    data and 0 is "good" data, this means the greedy mask would mask only
    those pixels that are "good" in all images, whereas the conservative
    mask would maks only those pixels that are "bad" in all images.
    Arguments:
        [masks ...] Any number of numpy.ndarrays, all of the same shape
        multiply    True to create a superposition of zeros (by multiplying masks together)
    '''
    base = np.zeros(masks[0].shape)
    if multiply:
        base = np.ones(masks[0].shape)

    for mask in masks:
        # Protect against someone, e.g., including True as a mask
        assert type(mask) == np.ndarray, 'Expected a numpy.ndarray type'

        if multiply:
            base = np.multiply(base, mask)

        else:
            base = np.where(mask > 0, mask, base)

    return base


def composite(reducers, *rasters, normalize='sum', nodata=-9999, dtype=np.float32):
    '''
    NOTE: Uses masked arrays in NumPy and therefore is MUCH slower than the
    `composite()` function, which is equivalent in output.

    Creates a multi-image (multi-date) composite from input rasters. The
    reducers argument specifies, in the order of the bands (endmembers), how
    to pick a value for that band in each pixel. If None is given, then the
    median value of that band from across the images is used for that pixel
    value. If None is specified as a reducer, the corresponding band(s) will
    be dropped. Combining None reducer(s) with a normalized sum effectively
    subtracts an endmember under the unity constraint. Arguments:
        reducers    One of ('min', 'max', 'mean', 'median', None) for each endmember
        rasters     One or more raster files to composite
        normalize   True (by default) to normalize results by their sum
        nodata      The NoData value (defaults to -9999)
        dtype       The data type to coerce in the output array; very important if the desired output is float but NoData value is integer
    '''
    shp = rasters[0].shape
    num_non_null_bands = shp[0] - len([b for b in reducers if b is None])
    assert all(map(lambda x: x == shp, [r.shape for r in rasters])), 'Rasters must have the same shape'
    assert len(reducers) == shp[0], 'Must provide a reducer for each band (including None to drop the band)'

    # Swap the sequence of rasters for a sequence of bands, then collapse the X-Y axes
    stack = np.array(rasters).swapaxes(0, 1).reshape(shp[0], len(rasters), shp[-1]*shp[-2])

    # Mask out NoData values
    stack_masked = np.ma.masked_where(stack == nodata, stack)

    # For each band (or endmember)...
    band_arrays = []
    for i in range(shp[0]):
        if reducers[i] in ('min', 'max', 'median', 'mean'):
            band_arrays.append(getattr(np.ma, reducers[i])(stack_masked[i, ...], axis=0))

    # Stack each reduced band (and reshape to multi-band image)
    final_stack = np.ma.vstack(band_arrays).reshape((num_non_null_bands, shp[-2], shp[-1]))

    # Calculate a normalized sum (e.g., fractions must sum to one)
    if normalize is not None:
        constant = getattr(final_stack, normalize)(axis=0) # The sum across the bands
        constant.set_fill_value(1.0) # NaNs will be divided by 1.0
        constant = np.ma.repeat(constant, num_non_null_bands, axis=0).reshape(final_stack.shape)
        # Divide the values in each band by the normalized sum across the bands
        if num_non_null_bands > 1:
            final_stack = final_stack / constant.swapaxes(0, 1)

        else:
            final_stack = final_stack / constant

    # NOTE: Essential to cast type, e.g., to float in case first pixel (i.e. top-left) is all NoData of an integer type
    final_stack.set_fill_value(dtype(nodata)) # Fill NoData for NaNs

    return final_stack.filled()


def composite2(reducers, *rasters, normalize='sum', nodata=-9999, dtype=np.float32):
    '''
    Creates a multi-image (multi-date) composite from input rasters. The
    reducers argument specifies, in the order of the bands (endmembers), how
    to pick a value for that band in each pixel. If None is given, then the
    median value of that band from across the images is used for that pixel
    value. Note that 'min,' 'max,' or 'mean' are each considerably faster than
    median or the starred versions because these methods are available on the
    NumPy array and do not require `apply_along_axis()` to be called. The
    median, 'min*' and 'max*' and 'mean*' reducers will remove NoData values
    before calculation. If None is specified as a reducer, the corresponding
    band(s) will be dropped. Combining None reducer(s) with a normalized sum
    effectively subtracts an endmember under the unity constraint. Arguments:
        reducers    One of ('min', 'max', 'mean', 'min*', 'max*', 'mean*', 'median', None) for each endmember
        rasters     One or more raster files to composite
        normalize   True (by default) to normalize results by their sum
        nodata      The NoData value (defaults to -9999)
        dtype       The data type to coerce in the output array; very important if the desired output is float but NoData value is integer
    '''
    shp = rasters[0].shape
    assert all(map(lambda x: x == shp, [r.shape for r in rasters])), 'Rasters must have the same shape'
    assert len(reducers) == shp[0] or len(reducers) == len(shp) - 1, 'Must provide a reducer for each endmember (including None for a median reduction)'

    # For single-band rasters...
    if len(shp) < 3:
        shp = (1, shp[0], shp[1])
        rasters = list(map(lambda r: r.reshape(shp), rasters))

    # For each band (or endmember)...
    num_non_null_bands = shp[0] - len([b for b in reducers if b is None])
    band_arrays = []
    for i in range(shp[0]):
        # Stack the rasters in a continuous, single-band "tapestry" using vstack(), then cut out the rasters concatenated in this way into separate bands using reshape()
        stack = np.vstack(map(lambda r: r[i,...], rasters))\
            .reshape((len(rasters), shp[1], shp[2]))

        # Get the reducing function for this band and apply across images
        if reducers[i] in ('min', 'max', 'mean'):
            band_arrays.append(getattr(stack, reducers[i])(axis=0))
            continue # Do not apply the more complex reducer_func below

        elif reducers[i] in ('min*', 'max*', 'mean*', 'median'):
            reducer_func_name = reducers[i].strip('*')

        # For None, skip this band (effectively subtracting it in a normalized sum)
        else:
            continue

        # If not all values are NoData, remove the NoData values before
        #   calculating the median; otherwise, the answer is NoData
        reducer_func = lambda xy: getattr(np, reducer_func_name)([
            a for a in xy if a != nodata
        ]) if not np.where(xy == nodata, True, False).all() else dtype(nodata)
        # NOTE: Essential to cast type, e.g., to float in case first pixel (i.e. top-left) is all NoData of an integer type
        band_arrays.append(np.apply_along_axis(reducer_func, 0, stack))

    # Stack each reduced band (and reshape to multi-band image)
    final_stack = np.vstack(band_arrays).reshape((num_non_null_bands, shp[-2], shp[-1]))

    # Calculate a normalized sum/ mean/ etc. (e.g., fractions must sum to one)
    if normalize is not None:
        # Again, essential to call dtype() to avoid erroneous type coercion
        normal_func = lambda x: x / getattr(x, normalize)() if not np.where(x == nodata, True, False).all() else np.repeat(dtype(nodata), num_non_null_bands)
        return np.apply_along_axis(normal_func, 0, final_stack)

    return final_stack


def density_slice(rast, rel=np.less_equal, threshold=1000):
    '''
    Returns a density slice from a given raster. Arguments:
        rast        A gdal.Dataset or a NumPy array
        rel         A NumPy logic function; defaults to np.less_equal
        threshold   An integer number
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        rastr = rast.ReadAsArray()

    else:
        rastr = rast.copy()

    if (len(rastr.shape) > 2 and min(rastr.shape) > 1):
        raise ValueError('Expected a single-band raster array')

    return rel(rastr, np.ones(rast.shape) * threshold).astype(np.int16)


def dump_raster(rast, rast_path, xoff=0, yoff=0, driver='GTiff', nodata=None):
    '''
    Creates a raster file from a given GDAL dataset (raster). Arguments:
        rast        A gdal.Dataset; does NOT accept NumPy array
        rast_path   The path of the output raster file
        xoff        Offset in the x-direction; should be provided when clipped
        yoff        Offset in the y-direction; should be provided when clipped
        driver      The name of the GDAL driver to use (determines file type)
        nodata      The NoData value; defaults to -9999.
    '''
    driver = gdal.GetDriverByName(driver)
    sink = driver.Create(rast_path, rast.RasterXSize, rast.RasterYSize,
        rast.RasterCount, rast.GetRasterBand(1).DataType)
    assert sink is not None, 'Cannot create dataset; there may be a problem with the output path you specified'
    sink.SetGeoTransform(rast.GetGeoTransform())
    sink.SetProjection(rast.GetProjection())

    for b in range(1, rast.RasterCount + 1):
        dat = rast.GetRasterBand(b).ReadAsArray()
        sink.GetRasterBand(b).WriteArray(dat)
        sink.GetRasterBand(b).SetStatistics(*map(np.float64,
            [dat.min(), dat.max(), dat.mean(), dat.std()]))

        if nodata is None:
            nodata = rast.GetRasterBand(b).GetNoDataValue()

            if nodata is None:
                nodata = -9999

        sink.GetRasterBand(b).SetNoDataValue(np.float64(nodata))

    sink.FlushCache()


def mae(reference, predictions, idx=None, n=1):
    '''
    Mean absolute error (MAE) for (p x n) raster arrays, where p is the number
    of bands and n is the number of pixels. Arguments:
        reference   Raster array of reference ("truth" or measured) data
        predictions Raster array of predictions
        idx         Optional array of indices at which to sample the arrays
        n           A normalizing constant for residuals; e.g., the number
                    of endmembers when calculating RMSE for modeled reflectance
    '''
    if idx is None:
        r = reference.shape[1]
        residuals = reference - predictions

    else:
        r = len(idx)
        residuals = reference[:, idx] - predictions[:, idx]

    # Divide the MSE by the number of bands before taking the root
    return np.apply_along_axis(lambda x: np.divide(np.abs(x).sum(), n), 0,
            residuals)


def mask_by_query(rast, query, invert=False, nodata=-9999):
    '''
    Mask pixels (across bands) that match a query in any one band or all bands.
    For example: `query = rast[1,...] < -25` queries those pixels with a value
    less than -25 in band 2; these pixels would be masked (if `invert=False`).
    By default, the pixels that are queried are masked, but if `invert=True`,
    the query serves to select pixels NOT to be masked (`np.invert()` can also
    be called on the query before calling this function to achieve the same
    effect). Arguments:
        rast    A gdal.Dataset or numpy.array instance
        query   A NumPy boolean array representing the result of a query
        invert  True to invert the query
        nodata  The NoData value to apply in the masking
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        rastr = rast.ReadAsArray()

    else:
        rastr = rast.copy()

    shp = rastr.shape
    if query.shape != rastr.shape:
        assert len(query.shape) == 2 or len(query.shape) == len(shp), 'Query must either be 2-dimensional (single-band) or have a dimensionality equal to the raster array'
        assert shp[-2] == query.shape[-2] and shp[-1] == query.shape[-1], 'Raster and query must be conformable arrays in two dimensions (must have the same extent)'

        # Transform the query into a 1-band array and then into a multi-band array
        query = query.reshape((1, shp[-2], shp[-1])).repeat(shp[0], axis=0)

    # Mask out areas that match the query
    if invert:
        rastr[np.invert(query)] = nodata

    else:
        rastr[query] = nodata

    return rastr


def mask_ledaps_qa(rast, mask, nodata=-9999):
    '''
    Applies a given LEDAPS QA mask to a raster. It's unclear how these
    bit-packed QA values ought to be converted back into 16-bit binary numbers:

    "{0:b}".format(42).zfill(16) # Convert binary to decimal padded left?
    "{0:b}".format(42).ljust(16, '0') # Or convert ... padded right?

    The temporary solution is to use the most common (modal) value as the
    "clear" pixel classification and discard everything else. We'd like to
    just discard pixels above a certain value knowing that everything above
    this threshold has a certain bit-packed QA meanining. For example, mask
    pixels with QA values greater than or equal to 12287:

    int("1000000000000000", 2) == 32768 # Maybe clouds
    int("0010000000000000", 2) == 12287 # Maybe cirrus

    Similarly, we'd like to discard pixels at or below 4, as these small binary
    numbers correspond to dropped frames, desginated fill values, and/or
    terrain occlusion. Arguments:
        rast    A gdal.Dataset or a NumPy array
        mask    A gdal.Dataset or a NumPy array
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        rast = rast.ReadAsArray()

    else:
        rastr = rast.copy()

    if not isinstance(mask, np.ndarray):
        maskr = mask.ReadAsArray()

    else:
        maskr = mask.copy()

    # Since the QA output is so unreliable (e.g., clouds are called water),
    #   we take the most common QA bit-packed value and assume it refers to
    #   the "okay" pixels
    mode = np.argmax(np.bincount(maskr.ravel()))
    assert mode > 4 and mode < 12287, "The modal value corresponds to a known error value"
    maskr[np.isnan(maskr)] = 0
    maskr[maskr != mode] = 0
    maskr[maskr == mode] = 1

    # Transform into a "1-band" array and apply the mask
    maskr = maskr.reshape((1, maskr.shape[0], maskr.shape[1]))\
        .repeat(rastr.shape[0], axis=0) # Copy the mask across the "bands"
    rastr[maskr == 0] = nodata
    return rastr


def mask_saturation(rast, saturation_value=10000, nodata=-9999):
    '''
    Masks out saturated values (e.g., surface reflectance values greater than
    16,000, however, SR values are only considered valid on the
    range [0, 10,000]). Arguments:
        rast                A gdal.Dataset or NumPy array
        saturation_value    The value beyond which pixels are considered saturated
        nodata              The NoData value; defaults to -9999.
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        rastr = rast.ReadAsArray()

    else:
        rastr = rast.copy()

    # Create a baseline "nothing is saturated in any band" raster
    mask = np.empty((1, rastr.shape[1], rastr.shape[2]))
    mask.fill(False)

    # Update the mask for saturation in any band
    for i in range(rastr.shape[0]):
        np.logical_or(mask, rastr[i,...] > saturation_value, out=mask)

    # Repeat the NoData value across the bands
    np.place(rastr, mask.repeat(rastr.shape[0], axis=0), (nodata,))
    return rastr


def pixel_to_geojson(pixel_pairs, gt=None, wkt=None, path=None, indent=2):
    '''
    This method translates given pixel locations into longitude-latitude
    locations on a given dataset. Assumes decimal degrees output. Arguments:
        pixel_pairs The pixel pairings to be translated in the form
                    [[x1, y1],[x2, y2]]
        gt          [Optional] A GDAL GeoTransform tuple
        wkt         [Optional] Projection information as Well-Known Text
        path        The file location of the GeoTIFF
    '''
    coords = pixel_to_xy(pixel_pairs, gt=gt, wkt=wkt, path=path, dd=True)
    doc = {
        'type': 'GeometryCollection',
        'geometries': []
    }
    for pair in coords:
        doc['geometries'].append({
            'type': 'Point',
            'coordinates': pair
        })

    return json.dumps(doc, sort_keys=False, indent=indent)


def pixel_to_xy(pixel_pairs, gt=None, wkt=None, path=None, dd=False):
    '''
    Modified from code by Zachary Bears (zacharybears.com/using-python-to-
    translate-latlon-locations-to-pixels-on-a-geotiff/).
    This method translates given pixel locations into longitude-latitude
    locations on a given GeoTIFF. Arguments:
        pixel_pairs The pixel pairings to be translated in the form
                    [[x1, y1],[x2, y2]]
        gt          [Optional] A GDAL GeoTransform tuple
        wkt         [Optional] Projection information as Well-Known Text
        path        The file location of the GeoTIFF
        dd          True to use decimal degrees for longitude-latitude (False
                    is the default)

    NOTE: This method does not take into account pixel size and assumes a
            high enough image resolution for pixel size to be insignificant.
    '''
    assert path is not None or (gt is not None and wkt is not None), \
        'Function requires either a reference dataset or a geotransform and projection'

    pixel_pairs = map(list, pixel_pairs)
    srs = osr.SpatialReference() # Create a spatial ref. for dataset

    if path is not None:
        ds = gdal.Open(path) # Load the image dataset
        gt = ds.GetGeoTransform() # Get geotransform of the dataset

        if dd:
            srs.ImportFromWkt(ds.GetProjection()) # Set up coord. transform.

    else:
        srs.ImportFromWkt(wkt)

    # Will use decimal-degrees if so specified
    if dd:
        ct = osr.CoordinateTransformation(srs, srs.CloneGeogCS())

    # Go through all the point pairs and translate them to pixel pairings
    ll_pairs = []
    for point in pixel_pairs:
        # Translate the pixel pairs into untranslated points
        lon = point[0] * gt[1] + gt[0]
        lat = point[1] * gt[5] + gt[3]
        if dd:
            (lon, lat, holder) = ct.TransformPoint(lon, lat) # Points to space

        ll_pairs.append((lon, lat)) # Add the point to our return array

    return ll_pairs


def rmse(reference, predictions, idx=None, n=1):
    '''
    RMSE for (p x n) raster arrays, where p is the number of bands and n is the
    number of pixels. RMSE is calculated after Powell et al. (2007) in Remote
    Sensing of the Environment. Arguments:
        reference   Raster array of reference ("truth" or measured) data
        predictions Raster array of predictions
        idx         Optional array of indices at which to sample the arrays
        n           A normalizing constant for residuals; e.g., the number
                    of endmembers when calculating RMSE for modeled reflectance
    '''
    shp = reference.shape
    if idx is None:
        residuals = (reference - predictions).reshape((1, shp[-2], shp[-1]))

    else:
        if len(reference.shape) == 3:
            residuals = reference.reshape((shp[0], shp[1]*shp[2]))[:, idx] - predictions.reshape((shp[0], shp[1]*shp[2]))[:, idx]

        else:
            residuals = reference[:, idx] - predictions[:, idx]

    # Divide the MSE by the number of bands before taking the root
    return np.sqrt(
        np.apply_along_axis(lambda x: np.divide(np.square(x).sum(), n), 0,
            residuals))


def spectra_at_idx(hsi_cube, idx):
    '''
    Returns the spectral profile of the pixels indicated by the indices
    provided. NOTE: Assumes an HSI cube (transpose of a GDAL raster).
    Arguments:
        hsi_cube    An HSI cube (n x m x p)
        idx         An array of indices that specify one or more pixels in a raster
    '''
    return np.array([hsi_cube[p[0],p[1],:] for p in idx])


def spectra_at_xy(rast, ll, gt=None, wkt=None, dd=False):
    '''
    Returns the spectral profile of the pixels indicated by the longitude-
    latitude pairs provided. Arguments:
        rast    A gdal.Dataset or NumPy array
        ll      An array of longitude-latitude pairs
        gt      A GDAL GeoTransform tuple; ignored for gdal.Dataset
        wkt     Well-Known Text projection information; ignored for gdal.Dataset
        dd      Interpret the longitude-latitude pairs as decimal degrees
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        gt = rast.GetGeoTransform()
        wkt = rast.GetProjection()
        rast = rast.ReadAsArray()

    # You would think that transposing the matrix can't be as fast as
    #   transposing the coordinate pairs, however, it is.
    return spectra_at_idx(rast.transpose(), xy_to_pixel(ll,
        gt=gt, wkt=wkt, dd=dd))


def stack_hdf_as_array(path, bands=(1,2,3,4,5,7), tpl=None):
    '''
    Stacks the layers of an HDF4 file assuming it contains subdata from EOS
    imagery (i.e., contains the prefix "HDF4_EOS:EOS_GRID:"); returns a tuple
    of (numpy.array, tuple) where the second argument is a GDAL GeoTransform.
    Assumes that the GeoTransform for all layers is the same (for downstream
    clipping purposes). The output array will have shape (b, m, n) where b is
    the number of bands. Arguments:
        path    The path to a GDAL dataset
        bands   A tuple of the band numbers to include
        tpl     The template for an HDF4 subdataset path
    '''
    if not os.path.exists(path):
        raise ValueError('File path not found')

    # Note: Template 'HDF4_EOS:EOS_GRID:"%s":Grid:band%d' is for LT5/LE7
    #   surface reflectance data
    if tpl is None:
        tpl = 'HDF4_EOS:EOS_GRID:"%s":Grid:sr_band%d'

    subdata = [tpl % (path, b) for b in bands]

    # Open the specified bands
    layers = [gdal.Open(n) for n in subdata]
    arr = np.array([l.GetRasterBand(1).ReadAsArray() for l in layers])
    gt = layers[0].GetGeoTransform()
    wkt = layers[0].GetProjection()
    for l in layers: l = None # Close layers
    return (arr, gt, wkt)


def stack_hdf_as_geotiff(path, dest=None, gt=None, wkt=None,
        bands=(1,2,3,4,5,7), preserve_band_nums=False):
    '''
    Stacks the layers of an HDF file as an output GeoTIFF; more flexible than
    stack_hdf_as_array() in that it will allow for non-EOS HDFs to be stacked.
    Assumes the first layer's spatial reference system is appropriate for all
    layers.
        path    The path to a GDAL dataset
        dest    The output destination file
        bands   A tuple of the band numbers to include
        gt      A GDAL GeoTransform
        wkt     Well-Known Text projection information
        preserve_band_nums  Should the band numbers be presered in output file?
    '''
    driver = gdal.GetDriverByName('GTiff')

    if not os.path.exists(path):
        raise ValueError('File path not found')

    # Create an output filename (destination)
    if dest is None:
        dest = list(os.path.basename(path).rpartition('.'))
        dest.insert(1, '_stack')
        dest[-1] = 'tiff'
        dest = ''.join(dest)

    hdf = gdal.Open(path)
    # Get the specified band names
    subdata = hdf.GetSubDatasets()

    # Match only certain band names
    if all(tuple(map(lambda x: type(x) is int, bands))):
        layers = []
        for b in bands:
            rx = re.compile(r'^HDF(4|5).*:Grid:.*band%d$' % b)
            layers.append(gdal.Open([
                s[0] for s in subdata if rx.match(s[0]) is not None
            ].pop()))

    else:
        rx = re.compile(r'^HDF(4|5).*:Grid:(' + '|'.join(bands) + ')$')
        bands = tuple(range(len(bands), 0, -1))
        layers = [gdal.Open(n) for n in [
            s[0] for s in subdata if regex.match(s[0]) is not None]
        ]

    hdf = None

    # Should band 7 always be band 7 (or band 6 if band 6 is missing)?
    n = len(bands)
    if preserve_band_nums:
        n = max(bands)

    # Figure out data type from the first layer, inherit GCS and projection
    sink = driver.Create(dest, layers[0].RasterXSize, layers[0].RasterYSize,
        n, layers[0].GetRasterBand(1).DataType)

    if gt is None:
        sink.SetGeoTransform(layers[0].GetGeoTransform())

    else:
        sink.SetGeoTransform(gt)

    if wkt is None:
        sink.SetProjection(layers[0].GetProjection())

    else:
        sink.SetProjection(wkt)

    for i, layer in enumerate(layers):
        b = i + 1
        if preserve_band_nums:
            b = bands[i]

        dat = layer.ReadAsArray()
        sink.GetRasterBand(b).WriteArray(dat)
        sink.GetRasterBand(b).SetStatistics(*map(np.float64,
            [dat.min(), dat.max(), dat.mean(), dat.std()]))

        # Set the NODATA value, if one is defined
        nodata = layer.GetRasterBand(1).GetNoDataValue()
        if nodata is not None:
            sink.GetRasterBand(b).SetNoDataValue(nodata)

    sink.FlushCache()
    return sink


def subarray(rast, filtered_value=-9999, indices=False):
    '''
    Given a (p x m x n) raster (or array), returns a (p x z) subarray where
    z is the number of cases (pixels) that do not contain the filtered value
    (in any band, in the case of a multi-band image). Arguments:
        rast            The input gdal.Dataset or a NumPy array
        filtered_value  The value to remove from the raster array
        indices         If True, return a tuple: (indices, subarray)
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        rastr = rast.ReadAsArray()

    else:
        rastr = rast.copy()

    shp = rastr.shape
    if len(shp) == 1:
        # If already raveled
        return rastr[rastr != filtered_value]

    if len(shp) == 2 or shp[0] == 1:
        # If a "single-band" image
        arr = rastr.reshape(1, shp[-2]*shp[-1])
        return arr[arr != filtered_value]

    # For multi-band images
    arr = rastr.reshape(shp[0], shp[1]*shp[2])
    idx = (arr != filtered_value).any(axis=0)
    if indices:
        # Return the indices as well
        rast_shp = (shp[-2], shp[-1])
        return (np.indices(rast_shp)[:,idx.reshape(rast_shp)], arr[:,idx])

    return arr[:,idx]


def xy_to_pixel(ll_pairs, gt=None, wkt=None, path=None, dd=False):
    '''
    Modified from code by Zachary Bears (zacharybears.com/using-python-to-
    translate-latlon-locations-to-pixels-on-a-geotiff/).
    This method translates given longitude-latitude pairs into pixel
    locations on a given GeoTIFF. Arguments:
        ll_pairs    The decimal lat/lon pairings to be translated in the
                    form [[lon1, lat1], [lon2, lat2]]
        gt          [Optional] A GDAL GeoTransform tuple
        wkt         [Optional] Projection information as Well-Known Text
        path        The file location of the GeoTIFF
        dd          True to use decimal degrees for longitude-latitude (False
                    is the default)

    NOTE: This method does not take into account pixel size and assumes a
            high enough image resolution for pixel size to be insignificant
    '''
    assert path is not None or (gt is not None and wkt is not None), \
        'Function requires either a reference dataset or a geotransform and projection'

    ll_pairs = map(list, ll_pairs)
    srs = osr.SpatialReference() # Create a spatial ref. for dataset

    if path is not None:
        ds = gdal.Open(path) # Load the image dataset
        gt = ds.GetGeoTransform() # Get geotransform of the dataset

        if dd:
            srs.ImportFromWkt(ds.GetProjection()) # Set up coord. transform.

    else:
        srs.ImportFromWkt(wkt) # Set up coord. transform.

    # Will use decimal-degrees if so specified
    if dd:
        ct = osr.CoordinateTransformation(srs.CloneGeogCS(), srs)

    # Go through all the point pairs and translate them to lng-lat pairs
    pixel_pairs = []
    for point in ll_pairs:
        if dd:
            # Change the point locations into the GeoTransform space
            (point[0], point[1], holder) = ct.TransformPoint(point[0], point[1])

        # Translate the x and y coordinates into pixel values
        x = (point[0] - gt[0]) / gt[1]
        y = (point[1] - gt[3]) / gt[5]
        pixel_pairs.append((int(x), int(y))) # Add point to our return array

    return pixel_pairs
