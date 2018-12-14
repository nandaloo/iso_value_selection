import itertools
import numpy as np
import pandas as pd


def grid_points(n):
    stepsize = 1.0/(n-1)
    x, y = np.mgrid[-1:1:stepsize, -1:1:stepsize]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    return pos


def reconstruct_dummy_x_y(pdf):
    """Given a sequence of density values construct x and y coordinate lists such that the the values in pdf
    could be the result of pdf(x,y). It expects that pdf is a square number.
    """
    n = np.int(np.sqrt(pdf.size))
    if not n*n == pdf.size:
        raise AssertionError("pdf must reshapable to a square matrix")
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)
    return (x, y)


def support_2d(pdf, range_x, n_x, range_y, n_y):
    x = np.linspace(range_x[0], range_x[1], n_x)
    y = np.linspace(range_y[0], range_y[1], n_y)
    z = [(xi, yi, pdf(xi, yi)) for (xi, yi) in itertools.product(x, y)]
    # z = [(i,j,pdf(i, j)) for i in x for j in y]
    return pd.DataFrame(z, columns=['x', 'y', 'pdf'])


def normalize_data(data):
    """Normalize data by translating and scaling it into the range [0,1] for all dimensions.

    Args:
        data  : 2d array
            array of arrays where the outer artray indexes attributes of the data.

    Returns:
        the normalized data as a 2d array
    """
    for d in range(data.shape[0]):
        values = data[d]
        min_, max_ = min(values), max(values)
        data[d] = (values - min_) / (max_ - min_)
    return data
