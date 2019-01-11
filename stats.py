"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de
"""

import numpy as np


def contour_level_weight(levels, data):
    """Compute the probability that levels occur, for given sequences of levels and densities.

    Note: This assumes a linearly spaced support!!

    Args:
        levels : sequence of numbers
            The contour level values in increasing order.

        data : np.array like
            The data to plot in a contour plot.
    """

    # probability weight per level
    sum_prob = np.zeros(len(levels) + 1)
    for p in data.flatten():
        idx = np.searchsorted(levels, p, side='left')
        sum_prob[idx] += p

    assert(len(sum_prob) == len(levels) + 1)

    return sum_prob


def embrace_ratio(levels, data):
    """Compute and return the embrace ratio w.r.t. to given levels.

    The embrace ratio is the percentage of weight embraced given levels. In other words, it's the ratio of
    sum(data) to sum(data point within contour levels).

    Args:
        levels : sequence of numbers
            Contour level values in increasing order.

        data : np.array like
            The data that `levels` refers to.

    Returns : scalar in range [0, 1]
        The embrace ratio.
    """

    data = np.asarray(data).flatten()
    level = levels[0]
    sum_outside = np.sum(data[data < level])
    sum_inside = np.sum(data[data >= level])

    return sum_inside / (sum_inside + sum_outside)


