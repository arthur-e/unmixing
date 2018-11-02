'''
Library for spatially adaptive spectral mixture analysis (SASMA).
A typical SASMA analysis is performed within a search window:

1. Set the NoData value in the endmember raster array to zero, such that any
   NoData areas do not contribute to a weighted sum.
2. Calculate the weight of each endmember in the search window, typically
   using inverse distance weighting (IDW).
'''

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


def kernel_idw_l1(size, band_num=None, normalize=True, moore_contiguity=False):
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
