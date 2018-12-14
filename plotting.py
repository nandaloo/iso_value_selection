"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de
"""

import numpy as np
import matplotlib.pyplot as plt

import iso_levels

# the plotly stuff is missing here

def plot_contour_levels_stat(levels_lst, pdf_lst, labels):
    # return that all contour levels have equal probability

    if (len(levels_lst) != len(pdf_lst) or len(levels_lst) != len(labels)):
        raise ValueError("All arguments must be sequences of equal length.")

    # k levels cut the prob values into k+1 subsets
    n = len(levels_lst)
    fig, ax = plt.subplots(figsize=(4 * n, 2), nrows=1, ncols=n)

    for i, (levels, pdf) in enumerate(zip(levels_lst, pdf_lst)):
        levels = levels.tolist()

        # probability weight per level
        sum_prob = np.zeros(len(levels) + 1)
        for p in pdf.flatten():
            idx = np.searchsorted(levels, p, side='left')
            sum_prob[idx] += p

        # print(sum_prob)
        # print(levels+[1])
        level_labels = ['L' + str(i + 1) for i in range(len(levels) + 1)]
        ax[i].bar(level_labels, sum_prob)
        ax[i].set_title(labels[i])


def contour_comparision_plot_2d(x, y, pdf, levels_lst, labels=['old', 'new']):
    """Plot a comparative plot of samples of a two dimensional quantitative function using two different
    contour level sets.

    Args:
        x : sequence of numbers
        y : sequence of numbers
            x and y together form the grid of input values where pdf provides function values.
        pdf : sequence of numbers
            values of the function to plot at all gridpoints of x and y
        levels_lst : a list of two lists of numbers.
            list of the contour levels.
    """
    n = len(levels_lst)
    pdf = pdf.reshape(len(x), len(y))

    # [levels_pdf, levels_prob] = levels_lst

    fig, ax = plt.subplots(figsize=(4 * n, 4), nrows=1, ncols=n)

    for i, (axis, label, levels) in enumerate(zip(ax, labels, levels_lst)):
        axis.contour(x, y, pdf, levels=levels)
        axis.set_title(label)

    #     ax[0].contour(x, y, pdf, levels=levels)
    #     ax[0].set_title(labels[0])

    #     ax[1].contour(x, y, pdf, levels=levels)
    #     ax[1].set_title(labels[1])

    plt.show()


def plot_combined(x, y, p, k=10):
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

    # plot stuff
    for i, ki in enumerate(k):
        plot_contour_levels_stat([pdf_lvls[i], prob_lvls[i], prob_hori_lvls[i]], [p, p ,p],
                                 [s.format(ki) for s in ['old (k={})', 'vertical (k={})', 'horizonal (k={})']])
        # ['old (k={})'.format(ki), 'vertical (k={})'.format(ki), 'horizonal (k={})'.format(ki)])
    for i, ki in enumerate(k):
        contour_comparision_plot_2d(x, y, p, [pdf_lvls[i], prob_lvls[i], prob_hori_lvls[i]],
                                    [s.format(ki) for s in ['old (k={})', 'vertical (k={})', 'horizonal (k={})']])
        # ['old (k={})'.format(ki), 'new (k={})'.format(ki)])