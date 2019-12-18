from distutils.core import setup

setup(name = 'unmixing',
    version = '0.2.4.dev',
    description = 'Interactive tools for spectral mixture analysis of multispectral raster data',
    author = 'K. Arthur Endsley',
    author_email = 'endsley@umich.edu',
    url = 'https://www.github.com/arthur-e/unmixing',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS'
    ],
    keywords = ['lsma spectral mixture analysis remote sensing raster'],
    packages = ['unmixing', 'unmixing.test'],
    py_modules = ['utils', 'lsma', 'sasma', 'transform', 'visualize', 'tests'],
    python_requires = '>=3',
    install_requires = [
        'numpy >= 1.8.2',
        'scipy >= 0.13.3',
        'matplotlib >= 1.3.1',
        'cvxopt >= 1.1.8',
        'GDAL >= 2.1.0',
        'pykml == 0.1.0',
        'pysptools >= 0.13.5',
        'Pillow >= 2.3.0',
        'scikit-learn >= 0.21.3'
    ],
    package_data = {
        'unmixing.test': [
            'clip_features.json',
            'FeatureSpace_selection_test.kml',
            'multi3_raster_clip.tiff',
            'multi3_raster.tiff',
            'multi7_mask.tiff',
            'multi7_raster_clip.tiff',
            'multi7_raster.tiff',
            'multi7_raster2.tiff',
            'LT05_020030_merge_19950712_stack_clip.tiff',
            'LT05_020030_merge_19950712_VBD_endmember_PIFs.tiff'
            ]
        }
    )
