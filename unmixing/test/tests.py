'''
This module contains the unit tests. Run "python tests.py" at the command line
to run the tests.
'''

import hashlib
import os
import random
import unittest
import unmixing
from unmixing.utils import *
from unmixing.lsma import FCLSAbundanceMapper, PPI, convex_hull_graham, endmembers_by_maximum_angle, endmembers_by_maximum_area, endmembers_by_query, endmembers_by_maximum_volume, hall_rectification, point_to_pixel_geometry
from unmixing.sasma import concat_endmember_arrays
from unmixing.transform import biophysical_composition_index, tasseled_cap_tm, mnf_rotation
from unmixing.visualize import FeatureSpace
from osgeo import gdal
from pysptools.noise import MNF

# For backwards compatibility in GDAL
gdal.SetConfigOption('GDAL_ARRAY_OPEN_BY_FILENAME', 'TRUE')
TEST_DIR = os.path.join(os.path.dirname(unmixing.__file__), 'test')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(unmixing.__file__)), 'docs/data')

class Tester(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        for p in [
            'temp.tiff', 'temp2.tiff', 'temp3.tiff', 'FeatureSpace_selection_test_1.kml',
            'rect_multi7_raster2.tiff'
        ]:
            try:
                os.unlink(os.path.join(self.test_dir, p))

            except FileNotFoundError:
                pass


class FCLS(Tester):
    test_dir = TEST_DIR
    data_dir = DATA_DIR

    def test_fcls_unmixing_with_single_endmember_spectra(self):
        '''
        Should calculate abundance from a mixed image for single endmember
        spectra.
        '''
        em_locs = [(326701, 4696895),(324978, 4699651), (328823, 4696835)]
        arr, gt, wkt = as_array(os.path.join(self.data_dir,
            'LT05_020030_merge_19950712_stack_clip.tiff'))
        endmembers = spectra_at_xy(mnf_rotation(arr).T, em_locs, gt, wkt)
        fcls_mapper = FCLSAbundanceMapper(arr[:,100:110,100:110],
            gt, wkt, processes = 1)
        result = fcls_mapper.map_abundance(endmembers)
        hasher = hashlib.sha256()
        hasher.update(result)
        self.assertEqual(hasher.hexdigest(), 'b042f3742910abd3505bf81a083eab6fc4684063ef30327f18d41327f2882b9f')

    def test_fcls_unmixing_with_single_endmember_spectra_multicore(self):
        '''
        Should calculate abundance from a mixed image for single endmember
        spectra; result should be the same for 1 or 2 processes.
        '''
        em_locs = [(326701, 4696895),(324978, 4699651), (328823, 4696835)]
        arr, gt, wkt = as_array(os.path.join(self.data_dir,
            'LT05_020030_merge_19950712_stack_clip.tiff'))
        endmembers = spectra_at_xy(mnf_rotation(arr).T, em_locs, gt, wkt)
        fcls_mapper1 = FCLSAbundanceMapper(arr[:,100:110,100:110],
            gt, wkt, processes = 1)
        fcls_mapper2 = FCLSAbundanceMapper(arr[:,100:110,100:110],
            gt, wkt, processes = 2)
        result1 = fcls_mapper1.map_abundance(endmembers)
        result2 = fcls_mapper2.map_abundance(endmembers)
        self.assertTrue(np.all(np.equal(result1, result2)))
        hasher = hashlib.sha256()
        hasher.update(result2)
        self.assertEqual(hasher.hexdigest(), 'b042f3742910abd3505bf81a083eab6fc4684063ef30327f18d41327f2882b9f')


class SASMA(Tester):
    test_dir = TEST_DIR
    data_dir = DATA_DIR

    def test_concatenation_of_endmember_arrays(self):
        '''
        Spectra arrays for multiple endmember types should be concatenated
        correctly; this step anticipates LSMA with multiple endmember spectra.
        '''
        arr, gt, wkt = as_array(os.path.join(self.data_dir,
            'LT05_020030_merge_19950712_stack_clip.tiff'))
        vbd, gt, wkt = as_array(os.path.join(self.data_dir,
            'LT05_020030_merge_19950712_VBD_endmember_PIFs.tiff'))
        emv = np.where(vbd == 1, arr, 0)
        emb = np.where(vbd == 2, arr, 0)
        emd = np.where(vbd == 3, arr, 0)
        endmembers = concat_endmember_arrays(emv, emb, emd)
        self.assertTrue(np.all(np.equal(endmembers.shape, (56639, 3, 6))))


class LSMA(Tester):
    test_dir = TEST_DIR
    cases = {
        'Vegetation': [
            (341663, 4709229), (314333, 4694229), (301583, 4697919),
            (320843, 4709769), (288053, 4741239)
        ],
        'High/Bright': [
            (331583, 4690839), (343313, 4707999), (351143, 4718739),
            (334913, 4709109), (305603, 4676919), (300683, 4707459),
            (318923, 4724259)
        ],
        'Low/Dark': [
            (325973, 4682799), (322193, 4678389), (321383, 4686279),
            (320033, 4694229), (334793, 4700829), (350393, 4727439)
        ]
    }

    def test_endmember_search_by_maximum_area(self):
        '''Should be able to search for endmembers by maximum area.'''
        rast, gt, wkt = as_array(os.path.join(self.test_dir, 'multi7_raster.tiff'))
        hsi_post_mnf = mnf_rotation(rast)

        # Test that dictionary type works
        result = endmembers_by_maximum_area(hsi_post_mnf.T, self.cases,
            gt=gt, wkt=wkt, dd=False)
        self.assertTrue(isinstance(result[0], np.ndarray))
        self.assertTrue(list(map(np.ceil, result[0][0].tolist())),
            [15.0, -9.0, 1.0])
        self.assertTrue(list(map(np.ceil, result[0][0].tolist())),
            [16.0, -8.0, 2.0])

        # Test that sequence/ array type works
        result = endmembers_by_maximum_area(hsi_post_mnf.T, self.cases['Vegetation'],
            gt=gt, wkt=wkt, dd=False)
        self.assertTrue(isinstance(result[0], np.ndarray))
        self.assertTrue(list(map(np.ceil, result[0][0].tolist())),
            [-12.0, 0.0, 0.0])
        self.assertTrue(list(map(np.ceil, result[0][0].tolist())),
            [-11.0, 1.0, 1.0])

    def test_endmember_search_by_maximum_volume(self):
        '''Should be able to search for endmembers by maximum volume.'''
        rast, gt, wkt = as_array(os.path.join(self.test_dir, 'multi7_raster.tiff'))
        hsi_post_mnf = mnf_rotation(rast)

        # Test that dictionary type works
        result = endmembers_by_maximum_volume(hsi_post_mnf.T, self.cases,
            gt=gt, wkt=wkt, dd=False)
        self.assertTrue(isinstance(result[0], np.ndarray))

        # Test that sequence/ array type works
        result = endmembers_by_maximum_volume(hsi_post_mnf.T, self.cases['Vegetation'],
            gt=gt, wkt=wkt, dd=False)
        self.assertTrue(isinstance(result[0], np.ndarray))

        ref_target = (300953, 4682679)
        result = endmembers_by_maximum_volume(hsi_post_mnf.T, self.cases['Vegetation'],
            ref_target, gt=gt, wkt=wkt, dd=False)
        self.assertTrue(isinstance(result[0], np.ndarray))

    def test_endmember_search_by_maximum_angle(self):
        '''Should be able to search for endmembers by maximum solid angle.'''
        rast, gt, wkt = as_array(os.path.join(self.test_dir, 'multi7_raster.tiff'))
        hsi = rast.transpose()
        hsi[hsi==-9999] = 0
        mnf = MNF()
        hsi_post_mnf = mnf.apply(hsi)

        # Test that sequence/ array type works
        ref_target = (342683, 4703919)
        result = endmembers_by_maximum_angle(hsi_post_mnf.T, self.cases['Vegetation'],
            ref_target, gt=gt, wkt=wkt, dd=False)
        self.assertTrue(isinstance(result[0], np.ndarray))

    def test_composite(self):
        '''Compositing multi-date images should produce the expected result.'''
        nodata = -9999
        ex1 = [
            np.array([
                [[-9999., 0.1], [-9999., -9999.]],
                [[-9999., 0.5], [-9999., -9999.]]
            ]),
            np.array([
                [[-9999., 0.9], [-9999., -9999.]],
                [[-9999., 0.2], [-9999., -9999.]]
            ]),
            np.array([
                [[-9999., 0.03], [-9999., -9999.]],
                [[-9999., 0.03], [-9999., -9999.]]
            ]),
            np.array([
                [[-9999., -9999.], [-9999., 0.2]],
                [[-9999., -9999.], [-9999., 0.2]]
            ]),
            np.array([
                [[-9999., 0.3], [0.4, 0.5]],
                [[-9999., 0.3], [0.4, 0.5]]
            ])
        ]

        # These tests use the function with masks that handles NoData correctly
        self.assertTrue(np.where(np.round(composite(('max', 'median'), *ex1, normalize=None), 2) == np.array([[[-9999., 0.9], [0.4, 0.5]], [[-9999., 0.25], [0.4, 0.35]]]), True, False).all())
        self.assertTrue(np.where(np.round(composite(('min', 'median'), *ex1, normalize=None), 2) == np.array([[[-9999., 0.03], [0.4, 0.2]], [[-9999., 0.25], [0.4, 0.35]]]), True, False).all())
        self.assertTrue(np.where(np.round(composite(('max', 'median'), *ex1, normalize=None, dtype=int), 2) == np.array([[[-9999., 0.9], [0.4, 0.5]], [[-9999., 0.25], [0.4, 0.35]]]), True, False).all()) # Setting `dtype=int` simulates the type coercion error we want to test

        # These tests use the function without masks that requires star functions
        self.assertTrue(np.where(np.round(composite2(('max', 'median'), *ex1, normalize=None), 2) == np.array([[[-9999., 0.9], [0.4, 0.5]], [[-9999., 0.25], [0.4, 0.35]]]), True, False).all())
        self.assertTrue(np.where(np.round(composite2(('min', 'median'), *ex1, normalize=None), 2) == np.array([[[-9999., -9999.], [-9999., -9999.]], [[-9999., 0.25], [0.4, 0.35]]]), True, False).all())
        self.assertTrue(np.where(np.round(composite2(('max', 'median'), *ex1, normalize=None, dtype=int), 2) == np.array([[[-9999., 0.9], [0.4, 0.5]], [[-9999., 0], [0, 0]]]), True, False).all()) # Setting `dtype=int` simulates the type coercion error we want to test

        # Should be able to subtract an endmember by setting None
        self.assertTrue(np.where(composite(('max', None), *ex1, normalize=None) == np.array([[[-9999., 0.9], [0.4, 0.5]]]), True, False).all())
        self.assertTrue(np.where(composite2(('max', None), *ex1, normalize=None) == np.array([[[-9999., 0.9], [0.4, 0.5]]]), True, False).all())

        # When subtracting one endmember from two and normalizing, all non-null
        #   pixels should sum to one
        self.assertTrue(np.where(composite(('max', None), *ex1, normalize='sum') == np.array([[[-9999., 1], [1, 1]]]), True, False).all())
        self.assertTrue(np.where(composite2(('max', None), *ex1, normalize='sum') == np.array([[[-9999., 1], [1, 1]]]), True, False).all())

    def test_convex_hull_graham(self):
        '''Should correctly calculate a convex hull.'''
        result = convex_hull_graham(self.cases['High/Bright'])
        self.assertEqual(result, [
            (300683, 4707459), (305603, 4676919), (331583, 4690839),
            (351143, 4718739),
            (318923, 4724259)
        ])

    def test_hall_rectification(self):
        '''Should rectify an image in the expected way.'''
        control_sets = {
            'High/Bright': [(331501.45,4694346.66), (319495.39,4706820.66), (298527.006,4691417.99)],
            'Low/Dark': [(322577.40,4658508.99), (361612.79,4694665.62), (378823.69,4692132.56)]
        }
        ref_raster = gdal.Open(os.path.join(self.test_dir, 'multi7_raster.tiff'))
        sub_raster = gdal.Open(os.path.join(self.test_dir, 'multi7_raster2.tiff'))

        # NOTE: Using same control sets for reference, subject
        hall_rectification(ref_raster, sub_raster, self.test_dir, control_sets, control_sets)

        arr, gt, wkt = as_array(os.path.join(self.test_dir, 'rect_multi7_raster2.tiff'))
        self.assertTrue(np.array_equal(arr.shape, (6, 74, 81)))
        self.assertTrue(np.array_equiv(arr[:,50,50].round(5), np.array([
            17, 1331, 1442, 3422, 2916, 2708
        ]).round(5)))

    def test_point_to_pixel_geometry(self):
        '''Should correctly calculate pixel geometry from center points.'''
        ds = gdal.Open(os.path.join(self.test_dir, 'multi3_raster.tiff'))
        xy_coords = pixel_to_xy([[10, 10], [20, 20]], ds.GetGeoTransform(),
            ds.GetProjection())
        geometries = point_to_pixel_geometry(xy_coords, source_epsg = 32617, target_epsg=4326)

        self.assertTrue(isinstance(geometries[0], ogr.Geometry))
        self.assertEqual(geometries[0].ExportToWkt(), 'POLYGON ((-84.8920166606528 42.4576139546101 0,-84.8916526360905 42.4576263335341 0,-84.8916359196253 42.457356779353 0,-84.8919999426335 42.4573444005452 0,-84.8920166606528 42.4576139546101 0))')
        self.assertEqual(geometries[1].ExportToWkt(), 'POLYGON ((-84.5128222586605 42.1997431501765 0,-84.5124595623182 42.1997542699897 0,-84.5124446081078 42.1994845879707 0,-84.5128073029137 42.1994734682621 0,-84.5128222586605 42.1997431501765 0))')

    def test_principle(self):
        '''
        Demonstrate the critical error that arises in composite() if the
        `dtype` argument is not set to `np.float32` when that kind of data is
        present.
        '''
        nodata = -9999

        # A 5-band array where the first pixel is -9999 in each band
        ex1 = np.array([
            [[ -9999, -9999],
             [ -9999, -9999]],
            [[ -9999, -9999],
             [ -9999, -9999]],
            [[ -9999, 3.18850607e-01],
             [ -9999, -9999]],
            [[ -9999, -9999],
             [ -9999, 2.08578259e-01]],
            [[ -9999, 2.99650848e-01],
             [ 3.32257748e-01, 4.80180055e-01]]])

        # A 5-band array where the first pixel is NOT -9999 in each band
        ex2 = np.array([
            [[ -9999, -9999],
             [ -9999, -9999]],
            [[ -9999, -9999],
             [ -9999, -9999]],
            [[ -9999, -9999],
             [ -9999, -9999]],
            [[ -9999, -9999],
             [ -9999, 2.08578259e-01]],
            [[ 3.18850607e-01, 2.99650848e-01],
             [ 3.32257748e-01, 4.80180055e-01]]])

        # Array with integer-valued -9999; causes erroneous behavior
        ex1a_out = np.round(np.apply_along_axis(lambda xy: np.median([
            a for a in xy if a != nodata
        ]) if not np.where(xy == nodata, True, False).all() else -9999,
            0, ex1), 2)
        self.assertTrue(np.where(ex1a_out == np.array([
            [-9.999e+03, 0],
            [         0, 0]]), True, False).all())

        # Array with integer-valued -9999 but doesn't cause error
        ex2a_out = np.round(np.apply_along_axis(lambda xy: np.median([
            a for a in xy if a != nodata
        ]) if not np.where(xy == nodata, True, False).all() else -9999,
            0, ex2), 3)
        self.assertTrue(np.where(ex2a_out == np.array([
            [ 0.319,  0.3  ],
            [ 0.332,  0.344]]), True, False).all())

        # Float-valued -9999.0 fixes error
        ex1b_out = np.round(np.apply_along_axis(lambda xy: np.median([
            a for a in xy if a != nodata
        ]) if not np.where(xy == nodata, True, False).all() else -9999.0,
            0, ex1), 2)
        self.assertTrue(np.where(ex1b_out == np.array([
            [-9.999e+03, 3.1e-01],
            [ 3.300e-01, 3.4e-01]]), True, False).all())

        ex2b_out = np.round(np.apply_along_axis(lambda xy: np.median([
            a for a in xy if a != nodata
        ]) if not np.where(xy == nodata, True, False).all() else -9999.0,
            0, ex2), 3)
        self.assertTrue(np.where(ex2b_out == np.array([
            [ 0.319,  0.3  ],
            [ 0.332,  0.344]]), True, False).all())

    def test_mnf_rotation(self):
        '''MNF rotation should achieve an expected result.'''
        file_path = os.path.join(self.test_dir, 'multi7_raster.tiff')
        ds = gdal.Open(file_path)
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        hsi = ds.ReadAsArray().transpose()
        hsi[hsi==nodata] = 0
        mnf = MNF()
        hsi_post_mnf = mnf.apply(hsi)
        self.assertEqual(int(hsi_post_mnf.diagonal()[0].sum()), 132)

    def test_ppi_endmember_extraction(self):
        '''
        Extraction by PPI, following an MNF rotation, should achieve an
        expected result.
        '''
        file_path = os.path.join(self.test_dir, 'multi7_raster.tiff')
        ds = gdal.Open(file_path)
        ppi = PPI()
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        hsi = ds.ReadAsArray().transpose()
        hsi[hsi==nodata] = 0
        mnf = MNF()
        hsi_post_mnf = mnf.apply(hsi)
        members = ppi.extract(hsi_post_mnf[...,0:3], 3, numSkewers=10000)
        self.assertEqual(int(members.sum()), 934)


class Transforms(Tester):
    test_dir = TEST_DIR

    def test_tasseled_cap_transformation(self):
        '''
        Should apply the tasseled cap transform and produce expected output.
        '''
        in_fname = os.path.join(self.test_dir, 'multi7_raster.tiff')
        out_fname = os.path.join(self.test_dir, 'temp.tiff')

        # Load the array; apply the transform
        rast, gt, wkt = as_array(in_fname)
        tm_image = tasseled_cap_tm(rast, ncomp = 6)
        self.assertEqual(tm_image.shape, (6, 74, 81))
        self.assertEqual(tm_image.mean().round(0), 212.0)
        self.assertTrue((tm_image[:,1,1] == np.array([0, 0, 0, 0, 0, 0])).all())
        self.assertTrue((tm_image[:,50,50].round(0) == np.array([ 3269.,  2328., -1752., -1140., 407., 18.])).all())

    def test_bci_calculation(self):
        '''
        Should calculate the biophysical composition index and produce
        expected output.
        '''
        in_fname = os.path.join(self.test_dir, 'multi7_raster.tiff')
        out_fname = os.path.join(self.test_dir, 'temp2.tiff')

        # Load the array; apply the transform
        rast, gt, wkt = as_array(in_fname)
        bci_image = biophysical_composition_index(rast, tc_func=tasseled_cap_tm)
        self.assertEqual(bci_image.shape, (1, 74, 81))
        self.assertEqual((bci_image.mean() * 100).round(0), -65.0)
        self.assertTrue((bci_image[:,1,1].round(3) == np.array([-0.574])).all())
        self.assertTrue((bci_image[:,50,50].round(3) == np.array([-0.702])).all())


class Utilities(Tester):

    test_dir = TEST_DIR

    def test_combine_dicts(self):
        '''
        Combining dictionaries of lists should also be idempotent/ with no
        side effects.
        '''
        foo = {'a': [1, 2, 3], 'b': [1]}
        bar = {'a': [4], 'b': [2, 3]}
        self.assertEqual(combine_dicts(foo, bar), {
            'a': [1, 2, 3, 4],
            'b': [1, 2, 3]
        })
        self.assertEqual(foo, {'a': [1, 2, 3], 'b': [1]})
        self.assertEqual(bar, {'a': [4], 'b': [2, 3]})

    def test_file_raster_and_array_access(self):
        '''
        Tests that essential file reading and raster/array conversion utilities
        are working properly.
        '''
        from_as_array = as_array(os.path.join(self.test_dir, 'multi3_raster.tiff'))
        from_as_raster = as_raster(os.path.join(self.test_dir, 'multi3_raster.tiff'))
        self.assertTrue(len(from_as_array) == len(from_as_raster) == 3)
        self.assertTrue(isinstance(from_as_array[0], np.ndarray))
        self.assertTrue(isinstance(from_as_raster[0], gdal.Dataset))

    def test_array_to_raster_interface(self):
        '''
        The array_to_raster() and array_to_raster_clone functions should
        perform as expected.
        '''
        # First round
        ds = gdal.Open(os.path.join(self.test_dir, 'multi3_raster.tiff'))
        gt = ds.GetGeoTransform()
        wkt = ds.GetProjection()
        arr = ds.ReadAsArray()
        ds = None
        rast = array_to_raster(arr, gt, wkt)
        self.assertEqual(gt, rast.GetGeoTransform())
        self.assertEqual(wkt, rast.GetProjection())

        # Second round
        rast2 = array_to_raster_clone(arr, os.path.join(self.test_dir,
            'multi7_raster.tiff'))
        self.assertEqual(gt, rast2.GetGeoTransform())
        self.assertEqual(wkt, rast2.GetProjection())

    def test_binary_mask(self):
        '''
        Should successfully apply a binary mask to an image.
        '''
        out_fname = os.path.join(self.test_dir, 'temp.tiff')
        rast, gt, wkt = as_array(os.path.join(self.test_dir, 'multi7_raster.tiff'))
        masked = binary_mask(rast, density_slice(rast[4, ...]))
        self.assertEqual(masked.shape, (6, 74, 81))
        self.assertEqual(masked.mean().round(0), -3627.0)
        self.assertTrue((masked[:,15,19] == np.array([348, 576, 374, 3159, 2033, 2386])).all())

    def test_saturation_mask(self):
        '''
        Should successfully mask out saturated pixels.
        '''
        out_fname = os.path.join(self.test_dir, 'temp.tiff')
        rast, gt, wkt = as_array(os.path.join(self.test_dir, 'multi7_raster.tiff'))
        mask = saturation_mask(rast)
        masked = binary_mask(rast, mask)
        self.assertEqual(masked.shape, (6, 74, 81))
        self.assertEqual(masked.mean().round(0), 881.0)
        self.assertTrue((masked[:,15,19] == np.array([348, 576, 374, 3159, 2033, 2386])).all())

    def test_density_slicing(self):
        '''
        Density slicing with the density_slice() function should perform as
        expected.
        '''
        rast, gt, wkt = as_array(os.path.join(self.test_dir, 'multi7_raster.tiff'))
        r1 = density_slice(rast[4, ...])
        r2 = density_slice(rast[4, ...], threshold=500)
        r3 = density_slice(rast[4, ...], rel=np.greater_equal)
        self.assertEqual(r1.shape, (74, 81))
        self.assertEqual(r2.shape, (74, 81))
        self.assertEqual(r3.shape, (74, 81))
        self.assertEqual(np.count_nonzero(r1), 2675)
        self.assertEqual(np.count_nonzero(r2), 2642)
        self.assertEqual(np.count_nonzero(r3), 3319)

    def test_lat_lng_to_pixel_and_reverse(self):
        '''
        Conversions from map coordinates to pixel coordinates and vice-versa
        should not introduce pixel-position errors of greater than 1 pixel.
        '''
        coords = [(random.randint(1, 70),
            random.randint(1, 70)) for x in range(1, 10)]
        file_path = os.path.join(self.test_dir, 'multi3_raster.tiff')
        p2ll = pixel_to_xy(coords, path=file_path, dd=True)
        ll2p = xy_to_pixel(p2ll, path=file_path, dd=True)
        self.assertTrue(np.max(np.abs((np.array(ll2p) - np.array(coords)))) <= 1)

    def test_lat_lng_to_pixel_and_reverse_without_reference_dataset(self):
        '''
        Should accurately locate the lat-long coordinates of two pixel
        coordinate pairs.
        '''
        coords = ((-84.5983, 42.7256), (-85.0807, 41.1138))
        pixels = [(18, 0), (2, 59)]
        file_path = os.path.join(self.test_dir, 'multi3_raster.tiff')
        ds = gdal.Open(file_path)
        gt = ds.GetGeoTransform()
        wkt = ds.GetProjection()
        p2ll = pixel_to_xy(pixels, gt=gt, wkt=wkt, dd=True)
        ll2p = xy_to_pixel(coords, gt=gt, wkt=wkt, dd=True)
        self.assertEqual(ll2p, [(18, 0), (2, 59)])
        self.assertTrue(np.max(np.abs((np.array(p2ll) - np.array(coords)))) <= 1)

    def test_spectral_profile(self):
        '''
        Should correctly retrieve a spectral profile from a raster dataset.
        '''
        coords = ((-84.5983, 42.7256), (-85.0807, 41.1138))
        pixels = [(18, 0), (2, 59)]
        file_path = os.path.join(self.test_dir, 'multi3_raster.tiff')
        ds = gdal.Open(file_path)
        kwargs = {
            'gt': ds.GetGeoTransform(),
            'wkt': ds.GetProjection(),
            'dd': True
        }

        # The true spectral profile
        spectra = np.array([[237, 418, 325], [507, 616, 445]], dtype=np.int16)
        sp1 = spectra_at_xy(ds, coords, **kwargs)
        sp2 = spectra_at_xy(ds.ReadAsArray(), coords, **kwargs)
        sp3 = spectra_at_idx(ds.ReadAsArray().transpose(), pixels)
        self.assertEqual(spectra.tolist(), sp1.tolist())
        self.assertEqual(spectra.tolist(), sp2.tolist())
        self.assertEqual(spectra.tolist(), sp3.tolist())

    def test_masking(self):
        '''
        Masking should go on without a hitch and the result should be just
        as expected.
        '''
        file_path = os.path.join(self.test_dir, 'multi7_raster.tiff')
        ds = gdal.Open(file_path)
        raw_mask = gdal.Open(os.path.join(self.test_dir, 'multi7_mask.tiff'))
        mask = cfmask(raw_mask, nodata=-9999)
        masked = binary_mask(ds.ReadAsArray(), mask)
        self.assertEqual(ds.ReadAsArray().diagonal()[0,0], 0)
        self.assertEqual(masked.diagonal()[0,0], -9999)


class Visualize(Tester):
    test_dir = TEST_DIR

    def test_interactive_feature_space(self):
        '''Tests that interactive plotting works as expected.'''
        path = os.path.join(self.test_dir, 'multi7_raster.tiff')
        vis = FeatureSpace(path, mask = None, cut_dim = 2, transform = True,
            nodata = -9999, epsg = 32617, keyword = 'test')
        fig = vis.plot_feature_space(c = None, interact = True, hold = True)
        vis.on_reset()
        vis.x0 = 161.56909597769811
        vis.y0 = 140.10416666666663
        vis.x1 = 171.02747909199519
        vis.y1 = 133.83487654320987
        vis.on_draw(output_dir = self.test_dir)

        with open(os.path.join(self.test_dir, 'FeatureSpace_selection_test.kml'), 'r') as stream:
            ref_contents = stream.read()

        with open(os.path.join(self.test_dir, 'FeatureSpace_selection_test_1.kml'), 'r') as stream:
            file_contents = stream.read()

        self.assertEqual(ref_contents, file_contents)


if __name__ == '__main__':
    unittest.main()
