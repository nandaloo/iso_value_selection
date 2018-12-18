import itertools
import numpy as np
import pandas as pd


DEFAULT_N = 50  # default number of support points along one axis
DEFAULT_D = 2  # default number of dimensions when creating support indexes dimension


def crossing_indexes(levels, p):
    """Return the indexes of values in p that are the first to exceed each of the levels in levels."""
    p = np.sort(p.flatten())
    levels = np.sort(levels)
    return np.searchsorted(p, levels)


def normalize_to_indexes(indexes=None, x=None, y=None, z=None, shape=None, d=None, n=None, low=None, high=None, data=None):
    """Get index sequences x and y from a flexible set of parameters.

    Args:
        indexes : sequence of sequences.
        x : sequences of numbers, optional.
            The support points along x-dimension to query the function on.
        y : sequences of numbers, optional.
            The support points along y-dimension to query the function on.
        z : sequences of numbers, optional.
            The support points along z-dimension to query the function on.
        shape : the desired shape as a tuple (i, j, ...), optional.
        d : integer, optional
            The dimension of the index to create.
        n : integer, optional
            the number of desired support points, for both dimensions.
        low : sequence of two integers
            The lower bound along x and y dimensions.
        high : sequence of two integers
            The upper bound along x and y dimensions.
        data : 1d or 2d data array, optional
            1st dimension indexes dimensions, 2nd the items. If low and high are missing, they are inferred from this
            data array.

    Returns:
        A tuple (x, y) of indexes along x and y dimension, respectively.
    """
    if indexes is None and x is not None:
        indexes = [x]
        if y is not None:
            indexes.append(y)
            if z is not None:
                indexes.append(z)

    if indexes is not None:
        indexes = [np.asarray(i) for i in indexes]
        if not np.all(np.isfinite(indexes)):
            raise ValueError('Values in x, y, z and indices must be finite numbers')
        return indexes

    # normalize data to 2-d np array
    if data is not None:
        data = np.asarray(data)
        if len(data.shape) == 1:
            data.shape = (1, data.shape[0])

    # normalize shape and d
    if shape is None:
        if d is None:
            if data is None:
                d = DEFAULT_D
            else:
                d = data.shape[0]
        if n is None:
            n = DEFAULT_N
        shape = (n,)*d
    d = len(shape)

    # normalize low and high, i.e. bounds of the indexes
    if data is not None:
        #data = np.squeeze(np.asarray(data))
        #if len(data.shape) != len(shape):
        #    raise ValueError('shape and shape of data do not match.')

        if low is None:
            low = np.amin(data, axis=1)
        if high is None:
            high = np.amax(data, axis=1)

    if low is None:
        low = [0]*d
    elif len(low) != d:
        raise ValueError("length of `low` does not match the given dimension `d` or dimension of `data`")

    if high is None:
        high = [1]*d
    elif len(high) != d:
        raise ValueError("length of `high` does not match the given dimension `d` or dimension of `data`")

    assert len(low) == len(high)
    assert len(low) == d

    # generate indexes
    indexes = [np.linspace(l, h, n) for l, h, n in zip(low, high, shape)]

    assert np.all(np.isfinite(indexes))
    return indexes


def indexes_to_input_sequence(indexes):
    mesh = np.meshgrid(*indexes)

    # flatten all index sequences
    for m in mesh:
        m.shape = (-1)

    return np.stack(mesh, axis=1)

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
    return x, y


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
