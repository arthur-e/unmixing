'''
Library for spatially adaptive spectral mixture analysis (SASMA).
A typical SASMA analysis is performed within a search window:

1. Calculate the weight of each endmember in the search window,
   typically using inverse distance weighting (IDW).
'''

def eye(size):
    # Get the index of the center, on either axis
    c = int(np.floor(np.median(range(0, size))))
    eye_win = np.ones((size, size))
    eye_win[c,c] = 0
    return eye_win


def kernel_idw_l1(size, band_axis=True, normalize=True, moore_contiguity=False):
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
                raise NotImplementedError("Moore contiguity/ Rook's rule not supported")

    if band_axis:
        window = window.reshape((1, size, size))

    if normalize:
        # Constraint: Sum of all weights must be equal to one
        return window / np.sum(window)

    return window
