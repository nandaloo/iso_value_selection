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


