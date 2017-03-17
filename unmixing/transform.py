'''
This module contains image processing and transformation utilities such as the
Tasseled Cap transformation.
'''

import numpy as np
from pysptools.noise import MNF

def __tasseled_cap__(rast, r, offset, nodata):
    shp = rast.shape

    # Should a translation be performed to prevent negative values?
    #TODO Offset does not exclude the possibility of the NoData value
    if offset:
        f = np.ones(shp)
        for b in range(0, shp[0]):
            f[b, ...] = f[b, ...] * abs(rast[b, ...].min())

    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        x = rast.ReadAsArray().reshape(shp[0], shp[1]*shp[2])

    else:
        x = rast.reshape(shp[0], shp[1]*shp[2])

    if offset:
        return np.add(np.dot(r, x).reshape(shp), f)

    return np.dot(r, x).reshape(shp)


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


def biophysical_composition_index(rast, nodata=-9999):
    '''
    Calculates the biophysical composition index (BCI) of Deng and Wu (2012)
    in Remote Sensing of Environment 127. The input raster is expected to be
    a tasseled cap-transformed raster. The NoData value is assumed to be
    negative (could never be the maximum value in a band).

    . sh/transform_BCI.sh data/20150730/LE70200301999196EDC00_tc_Oakland.tiff data/20150730/LE70200301999196EDC00_mask_Oakland.tiff data/20150730/LE70200301999196EDC00_bci_Oakland.tiff
    '''
    shp = rast.shape

    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        x = rast.ReadAsArray().reshape(shp[0], shp[1]*shp[2])

    else:
        x = rast.reshape(shp[0], shp[1]*shp[2])

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


def tasseled_cap_oli(rast, offset=False, nodata=-9999):
    '''
    Applies the Tasseled Cap transformation for OLI data. Assumes that the
    OLI data are "at-sensor" or top-of-atmosphere (TOA) data and that the
    bands are ordered (2,3,4,5,6,7). The coefficients for at-sensor data
    come from Baig et al. (2014) in Remote Sensing Letters 5:5.
    '''
    r = np.array([ # See Baig et al. (2014), Table 2
        ( 0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872), # Brightness
        (-0.2941,-0.2430,-0.5424, 0.7276, 0.0713,-0.1608), # Greenness
        ( 0.1511, 0.1973, 0.3283, 0.3407,-0.7117,-0.4559), # Wetness
        (-0.8239, 0.0849, 0.4396,-0.0580, 0.2013,-0.2773),
        (-0.3294, 0.0557, 0.1056, 0.1855,-0.4349, 0.8085),
        ( 0.1079,-0.9023, 0.4119, 0.0575,-0.0259, 0.0252)
    ], dtype=np.float32)

    return __tasseled_cap__(rast, r, offset, nodata)


def tasseled_cap_tm(rast, reflectance=True, offset=False, nodata=-9999):
    '''
    Applies the Tasseled Cap transformation for TM data. Assumes that the TM
    data are TM reflectance data (i.e., Landsat Surface Reflectance). The
    coefficients for reflectance factor data are taken from Crist (1985) in
    Remote Sensing of Environment 17:302.
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
        raise NotImplemented('The Tasseled Cap transformation for count data (DNs) has not yet been implemented')

    return __tasseled_cap__(rast, r, offset, nodata)
