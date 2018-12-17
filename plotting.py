"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de
"""

import numpy as np
import matplotlib.pyplot as plt

import iso_levels
import stats

# the plotly stuff is missing here


DEFAULT_LABELS = ['old', 'vertical', 'horizontal']  # default strings to use for labelling the plots


def _validate_infer_indexes(pdf, x, y):
    """Validate and normalize the the input.

    Given an array of 2d function values over an assumed grid of points and potentially the grid points, it normalizes
    these, such that:

        x, y are a np.array of indexes for the grid on which pdf contains the function values.
        pdf is a 2d np.array with shape matching x and y

    Args:
        pdf : 1d or 2d array
                x : sequence of numbers, optional.
        y : sequence of numbers, optional.
            x and y together form the grid of input values where pdf provides function values.
            It must be either both, x and y be given, or none of them. If both are given it must match the shape of pdf
            or pdf must be linear, in which case the original shape of pdf is reconstructed from the lenght of x and y.
    """

    pdf = np.asarray(pdf)

    if x is None and y is None:
        if len(pdf.shape) == 2:
            pass

        elif len(pdf.shape) == 1:
            # pdf must be square-able
            s = int(pdf.shape[0] ** 0.5)
            if s ** 2 == pdf.shape[0]:
                pdf.shape = s, s
            else:
                raise ValueError('Cannot infer shape of flattened data array pdf since it does not have a square '
                                 'number of elements.')

        # reconstruct x and y from shape
        x = np.linspace(0, 1, s)
        y = np.linspace(0, 1, s)

    elif (x is None and y is not None) or (x is not None and y is None):
        raise TypeError('x and y must be both set or both unset')

    elif x is not None and y is not None:
        if pdf.size != len(x) * len(y):
            raise ValueError('shapes of x,y and pdf do not match')
        pdf.shape = len(x), len(y)

    assert pdf.shape[0] == len(x) and pdf.shape[1] == len(y)
    return pdf, x, y


def contour(p, x, y, ax=None):
    """Draw contour plot over p, x, y.

    Optionally you may provide an axis to use for drawing.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.contour(x, y, p)

def plot_contour_levels_stat(levels_lst, pdf_lst, labels=DEFAULT_LABELS):
    """

    Args:
        levels_lst : sequence of numbers.
            the sequence of contour levels.
        pdf_lst : sequence of data arrays.
            The data array to draw a contour plot for.
        labels : sequence of strings, optional.
            String to use for labelling the plot.

    Returns:
         Nothing.
    """

    # return that all contour levels have equal probability
    if len(levels_lst) != len(pdf_lst) or len(levels_lst) != len(labels):
        raise ValueError("All arguments must be sequences of equal length.")

    # k levels cut the prob values into k+1 subsets
    n = len(levels_lst)
    fig, ax = plt.subplots(figsize=(4 * n, 2), nrows=1, ncols=n)

    for i, (levels, pdf) in enumerate(zip(levels_lst, pdf_lst)):

        sum_prob = stats.contour_level_weight(levels, pdf)

        level_labels = ['L' + str(i + 1) for i in range(len(levels) + 1)]
        ax[i].bar(level_labels, sum_prob)
        ax[i].set_title(labels[i])


def contour_comparision_plot_2d(levels_lst, pdf, x=None, y=None, labels=DEFAULT_LABELS):
    """Create and show comparative contour plots of a two dimensional quantitative function using different contour
    level sets.

    Args:
        x : sequence of numbers, optional.
        y : sequence of numbers, optional.
            x and y together form the grid of input values where pdf provides function values.
            It must be either both, x and y be given, or none of them. If both are given it must match the shape of pdf
            or pdf must be linear, in which case the original shape of pdf is reconstructed from the lenght of x and y.
        pdf : one or two dimensional sequence of numbers
            values of the function to plot at all gridpoints of x and y. See also `x` and `y`.
        levels_lst : a list of list of numbers.
            list of the contour levels.
    """

    pdf, x, y = _validate_infer_indexes(pdf, x, y)

    n = len(levels_lst)
    fig, ax = plt.subplots(figsize=(4 * n, 4), nrows=1, ncols=n)

    for i, (axis, label, levels) in enumerate(zip(ax, labels, levels_lst)):
        print(levels)
        axis.contour(x, y, pdf, levels=levels)
        axis.set_title(label)

    plt.show()


def plot_combined(p, x=None, y=None, k=iso_levels.DEFAULT_K):
    """Plot both, contour level stats, and the actual contour plots."""

    p, x, y = _validate_infer_indexes(p, x, y)

    # generate levels
    try:
        pdf_lvls = []
        prob_lvls = []
        prob_hori_lvls = []
        for ki in k:
            pdf_lvls.append(iso_levels.equi_value(p, ki))
            prob_lvls.append(iso_levels.equi_prob_per_level(p, ki))
            prob_hori_lvls.append(iso_levels.equi_horizontal_prob_per_level(p, ki))
    except TypeError:
        pdf_lvls = [iso_levels.equi_value(p, k)]
        prob_lvls = [iso_levels.equi_prob_per_level(p, k)]
        prob_hori_lvls = [iso_levels.equi_horizontal_prob_per_level(p, k)]
        k = [k]

    # plot
    for i, ki in enumerate(k):
        levels = [pdf_lvls[i], prob_lvls[i], prob_hori_lvls[i]]
        plot_contour_levels_stat(levels,
                                 len(levels)*[p],
                                 [s.format(ki) for s in ['old (k={})', 'vertical (k={})', 'horizonal (k={})']])
    for i, ki in enumerate(k):
        contour_comparision_plot_2d([pdf_lvls[i], prob_lvls[i], prob_hori_lvls[i]],
                                    p,
                                    x, y,
                                    [s.format(ki) for s in ['old (k={})', 'vertical (k={})', 'horizonal (k={})']])
