import numpy as np


def _find_nearest_non_vectorized(value, ticks):
    """
    Return the index of the element in the ticks array
    that the value input is closest to.

    Args:
        value (float, int, np.ndarray): the value to be found;
        ticks (np.ndarray) : 1D array of the elements that the input is
            going to be compared to.

    Returns:
        The index of the first closest value.

    """
    idx = (np.abs(ticks - value)).argmin()

    if idx.size > 1:
        idx = idx[0]
    elif idx.size == 0:
        raise ValueError

    return idx


def find_nearest(arr, ticks):
    """
    Vectorized version of _find_nearest_non_vectorized.

    Args:
        arr (np.ndarray): the array of elements to be find the indexes;
        ticks (np.ndarray) : 1D array of the elements that the input is
            going to be compared to.

    Returns:
        The array of indexes closest the values in arr.

    """
    vectorized_nearest = np.vectorize(_find_nearest_non_vectorized, excluded=['ticks'])
    return vectorized_nearest(value=arr, ticks=ticks).squeeze()
