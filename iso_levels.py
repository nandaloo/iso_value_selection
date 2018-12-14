"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de

A collection of algorithms to derive contour levels.

All algorithms implement an interface that accepts at least two arguments:

Args:
    data : 1d or 2d array
        `data` contains only numbers >= 0
        `data[ix,iy]` holds the value of some function f at points x,y, where x and y are value of a homogeneous 2d grid
        indexed by ix, iy. This is the support is on linear, equi-distant grid points in both directions!

    k : integer
        the number of levels to return

Returns:
    sequence of k levels in increasing order.
"""

import numpy as np
import pandas as pd

DEFAULT_K=5


def _check_data(data):
    """Check for data consistency, format and other required properties."""
    data = np.array(data)
    if len(data.shape) > 1 and data.shape[0] != data.shape[1]:
        # currently we only allow square arrays
        raise ValueError("data must be square array")

    if len(data.shape) > 1 and data.shape[0] == len(data):
        data = data.flatten()

    if not np.all(np.isfinite(data)):
        raise ValueError("data values may only be positive numbers or zero, but not negative, nan or infinite.")
    return data


def _check_levels(levels):
    """Sanity check for levels."""
    assert(np.all(np.diff(levels) >= 0))


def equi_value(data, k=DEFAULT_K):
    """The standard algorithm to determine contour levels: chose levels equidistantly along the value axis.

    Args:
        data : see module description.
        k : integer
            the number of levels to return
        """
    data = data.flatten()
    return np.linspace(np.min(data), np.max(data), k+2)[1:-1]


def equi_prob_per_level(data, k=DEFAULT_K):
    """Return k contour levels for provided array-like density values such that the combined area between two adjacent
    levels in a contour plot covers an equal volume (probability) of the density function.

    See module level documention for the interface documentation.

    Note: This assumes that data is the support of a probability density function.
    """

    pdf = _check_data(data)

    # convert to normalized probabilities
    # this assumes a linearly selected support!
    prob = pdf / np.sum(pdf)

    # need common dataframe for sorting by 'prob'
    df = pd.DataFrame(data={'pdf': pdf, 'prob': prob})
    df = df.sort_values(by=['prob'])

    # get cummulative prob in that order
    df['cum_prob'] = np.cumsum(df.prob)
    if not np.isclose([1], df['cum_prob'][-1:]):
        raise AssertionError('normalization failed')

    # level targets
    level_targets = np.linspace(0, 1, k + 2)[1:-1]

    # indices of corresponding pdf values
    indices = df['cum_prob'].searchsorted(level_targets, side='left')

    # pdf values are the contour level values
    levels = df.pdf.iloc[indices].values

    if not np.all(levels[:-1] <= levels[1:]):
        raise AssertionError('levels not sorted increasingly: {}'.format(levels))

    return levels


def equi_horizontal_prob_per_level(data, k=DEFAULT_K):
    """Return `k` contour levels provided array-like density values such that
     the increase from one level to the next one covers an equal volume of the density function.

    This creates 'horizontal slices of equal volume'.

    See module level documention for the interface documentation.

    Note: This assumes that data is the support of a probability density function.
    """

    pdf = _check_data(data)

    # 1. get level targets
    level_targets = np.linspace(0, 1, k + 2)[1:-1]
    #     print(level_targets)

    # 2. order by density
    pdf_sorted = np.sort(pdf)
    #     print(pdf_sorted)

    # 3. get (unnormalized) weight for each slice (i.e. each increase in density)
    # this is: (increase in p compared to previous p)*(number of p values equal or larger than p)
    # get neighboring difference
    pdf_shift = np.roll(pdf_sorted, 1)
    pdf_shift[0] = 0
    assert (len(pdf_shift) == len(pdf_sorted))
    pdf_diff = pdf_sorted - pdf_shift
    #     print(pdf_diff)

    # get weight as above product
    prob = pdf_diff * np.arange(len(pdf), 0, -1)
    #     print(prob)
    #     print(len(prob))

    # 4. normalize all weights
    prob_norm = prob / np.sum(prob)
    #     print(prob_norm)

    # 5. get cumulative weights
    cum_prob = np.cumsum(prob_norm)
    if not np.isclose([1], cum_prob[-1:]):
        raise AssertionError('normalization failed')

    # 6. find sorted
    indices = np.searchsorted(cum_prob, level_targets, side='left')
    #     print(indices)

    # 7. get density levels
    level_values = pdf_sorted[indices]
    assert (np.all(np.diff(level_values) >= 0))

    return level_values


if __name__ == '__main__':
    pass

