'''
This module contains image processing and transformation utilities such as the
Tasseled Cap transformation.
'''

import numpy as np
from pysptools.noise import MNF

def __tasseled_cap__(rast, rt, offset, ncomp=3):
    shp = shp2 = rast.shape

    # Some transformation matrices are abbreviated (e.g., they
    #   describe only the first 3 TC components)
    if rast.shape[0] != rt.shape[0]:
        shp2 = (rt.shape[0], shp[1], shp[2])

    # Should a translation be performed to prevent negative values?
    # TODO Offset does not exclude the possibility of the NoData value
    if offset:
        f = np.ones(shp)
        for b in range(0, shp[0]):
            f[b, ...] = f[b, ...] * abs(rast[b, ...].min())

    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        x = rast.ReadAsArray().reshape(shp[0], shp[1]*shp[2])

    else:
        x = rast.reshape(shp[0], shp[1]*shp[2])

    # Apply the transformation matrix rt
    if offset:
        return np.add(np.dot(rt, x).reshape(shp2), f)[0:ncomp, ...]

    return np.dot(rt, x).reshape(shp2)[0:ncomp, ...]


def ndvi(rast, red_idx=2, nir_idx=3):
    '''
    Calculates the normalized difference vegetation index (NDVI). Arguments:
        rast    An input raster or NumPy array
        red_idx The index of the Red (visible) band
        nir_idx The index of the near-infrared (NIR) band
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        rastr = rast.ReadAsArray()

    else:
        rastr = rast.copy()

    #TODO Does not account for NoData values
    return np.divide(rastr[nir_idx,...] - rastr[red_idx,...],
        rastr[nir_idx,...] + rastr[red_idx,...])


def mnf_rotation(rast, nodata=-9999):
    '''
    Applies the MNF rotation to an HSI data cube. Arguments:
        rast    A NumPy raster array
        nodata  The NoData value
    '''
    hsi = rast.copy().transpose()
    hsi[hsi==nodata] = 0 # Remap any lingering NoData values

    # Apply the Minimum Noise Fraction (MNF) rotation
    mnf = MNF()
    hsi_post_mnf = mnf.apply(hsi)
    hsi = None
    return hsi_post_mnf


def tasseled_cap_oli(rast, offset=False, nodata=-9999, ncomp=3):
    '''
    Applies the Tasseled Cap transformation for OLI data. Assumes that the
    OLI data are "at-sensor" or top-of-atmosphere (TOA) data and that the
    bands are ordered (2,3,4,5,6,7). The coefficients for at-sensor data
    come from Baig et al. (2014) in Remote Sensing Letters 5:5. Arguments:
        rast        The raster to be transformed
        reflectance Are the raster values reflectances?
        offset      An optional fixed offset to prevent negative values in
                    the output
        nodata      The NoData value
        ncomp       The number of Tasseled Cap components to return
    '''
    r = np.array([ # See Baig et al. (2014), Table 2
        ( 0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872), # Brightness
        (-0.2941,-0.2430,-0.5424, 0.7276, 0.0713,-0.1608), # Greenness
        ( 0.1511, 0.1973, 0.3283, 0.3407,-0.7117,-0.4559), # Wetness
        (-0.8239, 0.0849, 0.4396,-0.0580, 0.2013,-0.2773),
        (-0.3294, 0.0557, 0.1056, 0.1855,-0.4349, 0.8085),
        ( 0.1079,-0.9023, 0.4119, 0.0575,-0.0259, 0.0252)
    ], dtype=np.float32)

    return __tasseled_cap__(rast, r, offset, ncomp)


def tasseled_cap_tm(rast, reflectance=True, offset=False, nodata=-9999,
        ncomp=3):
    '''
    Applies the Tasseled Cap transformation for TM data. Assumes that the TM
    data are TM reflectance data (i.e., Landsat Surface Reflectance). The
    coefficients for reflectance factor data are taken from Crist (1985) in
    Remote Sensing of Environment 17:302. The coefficients for DN data are
    taken from Liu et al. (2015) in Int. Journal of Remote Sensing, citing
    Crist et al. (1986). Arguments:
        rast        The raster to be transformed
        reflectance Are the raster values reflectances?
        offset      An optional fixed offset to prevent negative values in
                    the output
        nodata      The NoData value
        ncomp       The number of Tasseled Cap components to return
    '''
    if reflectance:
        # Reflectance factor coefficients for TM bands 1-5 and 7; they are
        #   entered here in tabular form so they are already transposed with
        #   respect to the form suggested by Kauth and Thomas (1976)
        r = np.array([ # See Crist (1985), Table 1
            ( 0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303), # Brightness
            (-0.1603,-0.2819,-0.4934, 0.7940,-0.0002,-0.1446), # Greenness
            ( 0.0315, 0.2021, 0.3102, 0.1594,-0.6806,-0.6109), # Wetness
            (-0.2117,-0.0284, 0.1302,-0.1007, 0.6529,-0.7078),
            (-0.8669,-0.1835, 0.3856, 0.0408,-0.1132, 0.2272),
            ( 0.3677,-0.8200, 0.4354, 0.0518,-0.0066,-0.0104)
        ], dtype=np.float32)

    else:
        r = np.array([ # Crist et al. (1986) as cited by Liu et al. (2015)
            ( 0.2909, 0.2493, 0.4806, 0.5568, 0.4438, 0.1706), # Brightness
            (-0.2728,-0.2174,-0.5508, 0.7220, 0.0733,-0.1648), # Greenness
            ( 0.1446, 0.1761, 0.3322, 0.3396,-0.6210, 0.4186)  # Wetness
        ])

    return __tasseled_cap__(rast, r, offset, ncomp)


def tasseled_cap_etm_plus(rast, toa=True, offset=False, nodata=-9999, ncomp=3):
    '''
    Applies the Tasseled Cap transformation for ETM+ data. The coefficients
    for come from Liu et al. (2016) in the Journal of Spatial Science.
    Coefficients available for either top-of-atmosphere (TOA) or digital
    number (DN) data.
    Arguments:
        rast        The raster to be transformed
        toa         Are the raster values (at-satellite) reflectances?
        offset      An optional fixed offset to prevent negative values in
                    the output
        nodata      The NoData value
        ncomp       The number of Tasseled Cap components to return
    '''
    if toa:
        # Reflectance factor coefficients for ETM+ bands 1-5 and 7; they are
        #   entered here in tabular form so they are already transposed with
        #   respect to the form suggested by Kauth and Thomas (1976)
        r = np.array([ # See Liu et al. (2016), Table 2
            ( 0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596),
            (-0.3344,-0.3544,-0.4556, 0.6966,-0.0242,-0.2630),
            ( 0.2626, 0.2141, 0.0926, 0.0656,-0.7629,-0.5388)
        ], dtype=np.float32)

    else:
        r = np.array([ # For digital number (DN) data
            ( 0.3627, 0.4005, 0.5216, 0.2600, 0.4279, 0.4304),
            (-0.0997, 0.0074,-0.1985, 0.9230, 0.0673,-0.3068),
            ( 0.4217, 0.3581, 0.3210,-0.0024,-0.6037,-0.4759)
        ], dtype=np.float32)

    return __tasseled_cap__(rast, r, offset, ncomp)


def biophysical_composition_index(rast, tc_func=tasseled_cap_tm, nodata=-9999):
    '''
    Calculates the biophysical composition index (BCI) of Deng and Wu (2012)
    in Remote Sensing of Environment 127. The NoData value is assumed to be
    negative (could never be the maximum value in a band). Arguments:
        rast    A NumPy Array or gdal.Dataset instance
        tc_func The function to be used to transform the input raster to
                Tasseled Cap brightness, greenness, and wetness
        nodata  The NoData value to ignore
    '''
    shp = rast.shape

    # Perform the tasseled cap rotation
    x = tc_func(rast, ncomp=3).reshape(3, shp[1]*shp[2])
    unit = np.ones((1, shp[1] * shp[2]))

    stack = []
    for i in range(0, 3):
        # Calculate the minimum values after excluding NoData values
        tcmin = np.setdiff1d(x[i, ...].ravel(), np.array([nodata])).min()
        stack.append(np.divide(np.subtract(x[i, ...], unit * tcmin),
            unit * (x[i, ...].max() - tcmin)))

    # Unpack the High-albedo, Vegetation, and Low-albedo components
    h, v, l = stack

    return np.divide(
        np.subtract(np.divide(np.add(h, l), unit * 2), v),
        np.add(np.divide(np.add(h, l), unit * 2), v))\
        .reshape((1, shp[1], shp[2]))


def rndsi(rast, tc_func=tasseled_cap_tm, bands=(6,2), nodata=-9999):
    '''
    Calculates the ratio normalized difference soil index (RNDSI) after
    Deng et al. (2015) in Int. J. of App. Earth Obs. and Geoinf. Arguments:
        rast    A NumPy Array or gdal.Dataset instance
        tc_func The function to be used to transform the input raster to
                Tasseled Cap brightness, greenness, and wetness
        bands   A tuple of integers, the SWIR2 band number and the Green band
                number, in that order
        nodata  The NoData value to ignore
    '''
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        arr = rast.ReadAsArray()

    else:
        arr = rast

    shp = arr.shape

    # Perform the tasseled cap rotation; obtain TC1
    tc1 = tc_func(arr, ncomp=1)
    tc1 = np.where(arr[0,...] == nodata, np.nan, tc1)

    # Calculate NDSI
    b_swir2, b_green = tuple(map(lambda x: x - 1, bands))
    ndsi = np.divide(
        np.subtract(arr[b_swir2,...], arr[b_green,...]),
        np.add(arr[b_swir2,...], arr[b_green,...])).reshape((1, shp[1], shp[2]))
    ndsi = np.where(arr[0,...] == nodata, np.nan, ndsi)

    # Calculate normalized TC1
    ntc1 = np.divide(
        np.subtract(tc1, np.nanmin(tc1)),
        np.subtract(np.nanmax(tc1), np.nanmin(tc1)))

    # Calculate normalized NNDSI
    nndsi = np.divide(
        np.subtract(ndsi, np.nanmin(ndsi)),
        np.subtract(np.nanmax(ndsi), np.nanmin(ndsi)))

    rndsi = np.divide(nndsi, np.where(ntc1 == 0, np.nan, ntc1))
    return np.where(rndsi == np.nan, nodata, rndsi)
