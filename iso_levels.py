"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de

A collection of algorithms to derive contour levels.
"""
import logging

import numpy as np
import pandas as pd


DEFAULT_K = 5  # default number of contour lines


def _validate_normalize_data(data):
    """Check for data consistency, format and other required properties."""
    data = np.asarray(data)
    if len(data.shape) > 1 and data.shape[0] != data.shape[1]:
        # currently we only allow square arrays
        raise ValueError("data must be square array")

    if len(data.shape) > 1 and data.shape[0] == len(data):
        data = data.flatten()

    if not np.all(np.isfinite(data)):
        raise ValueError("data values may only be positive numbers or zero, but not negative, nan or infinite.")

    return data


def _validate_normalize_levels(levels, k, remove_double_levels=False):
    """Sanity check for levels.

    Args:
        k : integer
            Number of levels to produce.
    """

    # must be ordered
    assert(np.all(np.diff(levels) >= 0)), "iso levels are not ordered"

    # must be the correct number
    assert(len(levels) == k), "produces only {} many iso levels".format(k)

    # but it may happen that values occur twice: remove them
    if remove_double_levels:
        levels = np.unique(levels)

    return levels


def _get_level_positions(max_value, equal_spaced_level: int=DEFAULT_K, custom_spaced_level: list=None):
    """ Calculate the position of level depending on if k is given or relative_level
                        when a list of values between 0 and 1 are given the levels are relative to those
                        when a number is given it spaces the level equidistant

    Args:
        max_value: max value of the target-distribution

        k: number of iso level when they are supposed to be equally distributed

        custom_spaced_level: relative positions of the level when they should be customized

    Returns:
        A sequence of level positions in increased order
    """
    if isinstance(custom_spaced_level, list):
        if min(custom_spaced_level) >= 0. and max(custom_spaced_level) <= 1.:
            return [x * max_value for x in custom_spaced_level]
        else:
            logging.warning(f"[{min(custom_spaced_level)}, ... ,{max(custom_spaced_level)}] not in 0-1. \n"
                            f"Setting iso level equal spaced: {equal_spaced_level}")
    return np.linspace(0, max_value, equal_spaced_level + 2)[1:-1]


def equi_value(data, k=DEFAULT_K, custom_spaced_level=None):
    """The standard algorithm to determine contour levels: chose levels equidistantly along the value axis.

    Args:
        data : 1d or 2d array
            data is the array of values to choose contour levels for. It may only hold numbers >= 0, and is the support
            of some function on linear, equi-distant grid points in both directions, i.e. `data[ix,iy]` holds the value
            of some function f at points x,y, where x and y are value of a homogeneous 2d grid indexed by ix, iy. This is

        k : integer
            The number of levels to return.

        custom_spaced_level : list
            A list of values between 0 and 1 which give a relative position of the iso level by distance. This somehow
            removes the logic of equi_value, but makes sense to me because I can do the same with the other methods.
            E.g. for equi_prob_per_level I can set custom probability distances between lines.

    Returns:
        A sequence of k levels in increasing order.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    data = _validate_normalize_data(data)
    #levels = np.linspace(np.min(data), np.max(data), k+2)[1:-1]
    levels = _get_level_positions(np.max(data), k, custom_spaced_level)
    return _validate_normalize_levels(levels, k)


def equi_prob_per_level(data, k=DEFAULT_K, custom_spaced_level=None):
    """Return k contour levels for provided array-like density values such that the combined area between two adjacent
    levels in a contour plot covers an equal volume (probability) of the density function.

    Note: This assumes that data is the support of a probability density function.

    Args:
        data : 1d or 2d array
            data is the array of values to choose contour levels for. It may only hold numbers >= 0, and is the support
            of some function on linear, equi-distant grid points in both directions, i.e. `data[ix,iy]` holds the value
            of some function f at points x,y, where x and y are value of a homogeneous 2d grid indexed by ix, iy. This is

        k : integer
            The number of levels to return.

        custom_spaced_level : list
            A list of values between 0 and 1 which give a relative position of the iso level by probability. This somehow
            removes the logic of equi_prob_per_level, but makes sense to me because I can do the same with the other
            methods.
            E.g. for equi_value I can set custom distances between lines.

    Returns:
        A sequence of k levels in increasing order.

    """
    if k < 1:
        raise ValueError("k must be at least 1")

    pdf = _validate_normalize_data(data)

    # convert to normalized probabilities
    # this assumes a linearly selected support!
    prob = pdf / np.sum(pdf)

    # need common data frame for sorting by 'prob'
    df = pd.DataFrame(data={'pdf': pdf, 'prob': prob})
    df = df.sort_values(by=['prob'])

    # get cumulative prob in that order
    df['cum_prob'] = np.cumsum(df.prob)
    if not np.isclose([1], df['cum_prob'][-1:]):
        raise AssertionError('normalization failed')

    # drop all rows with duplicate pdf values,
    # we keep the last occurrence, because we need the cumprob value that a given pdf limit "reaches"
    df = df.drop_duplicates(subset='pdf', keep='last')

    # level targets
    level_targets = _get_level_positions(1, k, custom_spaced_level)

    # indices of corresponding pdf values that make us exceed 1/k probability
    indices = df['cum_prob'].searchsorted(level_targets, side='left')

    # choose iso value as the mid value between the exceeding pdf value and the previous one (if exists)
    #  advantage: iso value never collides with actual existing pdf value
    indices_previous = [0 if i == 0 else i-1 for i in indices]

    # verify
    values = df.pdf.iloc[indices].values
    values_previous = df.pdf.iloc[indices_previous].values
    assert(np.all(np.logical_or(
        values_previous < values,
        values_previous == 0
    )))

    # pdf values are the contour level values
    levels = (values_previous + values)/2
    return _validate_normalize_levels(levels, k)


def equi_horizontal_prob_per_level(data, k=DEFAULT_K, custom_spaced_level=None):
    """Return `k` contour levels provided array-like density values such that
     the increase from one level to the next one covers an equal volume of the density function.

    This creates 'horizontal slices of equal volume'.

    Note: This assumes that data is the support of a probability density function.

    Args:
        data : 1d or 2d array
            data is the array of values to choose contour levels for. It may only hold numbers >= 0, and is the support
            of some function on linear, equi-distant grid points in both directions, i.e. `data[ix,iy]` holds the value
            of some function f at points x,y, where x and y are value of a homogeneous 2d grid indexed by ix, iy. This is

        k : integer
            The number of levels to return.

        custom_spaced_level : list
            A list of values between 0 and 1 which give a relative position of the iso level by probability. This somehow
            removes the logic of equi_prob_per_level, but makes sense to me because I can do the same with the other
            methods.
            E.g. for equi_value I can set custom distances between lines.

    Returns:
        A sequence of k levels in increasing order.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    pdf = _validate_normalize_data(data)

    # 1. get level targets
    level_targets = _get_level_positions(1, k, custom_spaced_level)

    # 2. order by density
    pdf_sorted = np.sort(pdf)

    # 3. get (unnormalized) weight for each slice (i.e. each increase in density)
    # this is: (increase in p compared to previous p)*(number of p values equal or larger than p)

    # get neighboring difference
    pdf_shift = np.roll(pdf_sorted, 1)
    pdf_shift[0] = 0
    assert (len(pdf_shift) == len(pdf_sorted))
    pdf_diff = pdf_sorted - pdf_shift

    # get weight as above product
    prob = pdf_diff * np.arange(len(pdf), 0, -1)

    # 4. normalize all weights
    prob_norm = prob / np.sum(prob)

    # 5. get cumulative weights
    cum_prob = np.cumsum(prob_norm)
    if not np.isclose([1], cum_prob[-1:]):
        raise AssertionError('normalization failed')

    # 6. find sorted
    indices = np.searchsorted(cum_prob, level_targets, side='left')

    # 7. get density levels
    levels = pdf_sorted[indices]
    return _validate_normalize_levels(levels, k)


if __name__ == '__main__':
    pass

