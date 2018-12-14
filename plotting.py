"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de
"""

import numpy as np
import matplotlib.pyplot as plt

import iso_levels
import stats

# the plotly stuff is missing here


DEFAULT_LABELS = ['old', 'vertical', 'horizontal']  # default strings to use for labelling the plots


def plot_contour_levels_stat(levels_lst, pdf_lst, labels=DEFAULT_LABELS):
    """

    Args:
        levels_lst : sequence of numbers.
            the sequence of contour levels.
        pdf_lst : sequence of data arrays.
            the data array to draw a contour plot for.
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
    """Plot a comparative plot of samples of a two dimensional quantitative function using two different
    contour level sets.

    Args:
        x : sequence of numbers, optional.
        y : sequence of numbers, optional.
            x and y together form the grid of input values where pdf provides function values.
            It must be either both, x and y be given, or none of them. If both are given it must match the shape of pdf.
        pdf : sequence of numbers
            values of the function to plot at all gridpoints of x and y
        levels_lst : a list of two lists of numbers.
            list of the contour levels.
    """
    n = len(levels_lst)

    pdf = np.asarray(pdf)

    if x is None and y is None:
        # pdf must be 2d or square
        if len(pdf.shape) == 2 and pdf.shape[0] == pdf.shape[1]:
            pass
        else:
            s = pdf.shape[0] ** 0.5
            if s ** 0.5 == int(s ** 0.5):
                pdf.reshape(s,s)
            else:
                raise ValueError('cannot infer shape of flattened data array pdf ')

    pdf = pdf.reshape(len(x), len(y))

    fig, ax = plt.subplots(figsize=(4 * n, 4), nrows=1, ncols=n)

    for i, (axis, label, levels) in enumerate(zip(ax, labels, levels_lst)):
        axis.contour(x, y, pdf, levels=levels)
        axis.set_title(label)

    plt.show()


def plot_combined(x, y, p, k=iso_levels.DEFAULT_K):
    """plot both, contour level stats, and the actual contour plots"""
    p = p.flatten()

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
        plot_contour_levels_stat([pdf_lvls[i], prob_lvls[i], prob_hori_lvls[i]], [p, p ,p],
                                 [s.format(ki) for s in ['old (k={})', 'vertical (k={})', 'horizonal (k={})']])
    for i, ki in enumerate(k):
        contour_comparision_plot_2d(x, y, p, [pdf_lvls[i], prob_lvls[i], prob_hori_lvls[i]],
                                    [s.format(ki) for s in ['old (k={})', 'vertical (k={})', 'horizonal (k={})']])
