'''
Contains a number of stored data structures and tools for working with Earth
Observing System (EOS) data. For example, contains commonly used values for
pixels to be masked from the CFMask layer.
'''

# Common CFMask values; see https://landsat.usgs.gov/landsat-surface-reflectance-quality-assessment
cfmask_values = {
    # Landsat 4-7 Pre-Collection pixel_qa values to be masked:
    'pre-collection': (1, 2, 3, 4, 255),
    'collection1': {
        # Landsat 8 Collection 1 pixel_qa values to be masked
        'landsat8': {
            'low+': (324, 328, 336, 352, 368, 386, 388, 392, 400, 416, 432, 480, 832, 836, 840, 848, 864, 880, 900, 904, 912, 928, 944, 992, 1024),
            'medium+': (324, 328, 386, 388, 392, 400, 416, 432, 480, 832, 836, 840, 848, 864, 880, 900, 904, 912, 928, 944, 992, 1024)
        },
        # Landsat 4-7 Collection 1 pixel_qa values to be masked
        'landsat4-7': {
            'medium+': (68, 72, 80, 112, 132, 136, 144, 160, 176, 224),
            'medium+_exclude_water': (72, 80, 112, 136, 144, 160, 176, 224)
        }
    }
}
