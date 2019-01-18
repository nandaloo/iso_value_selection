"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import iso_levels
import stats
import utils


DEFAULT_LABELS = ['old', 'vertical', 'horizontal']  # default strings to use for labelling the plots

cfg = {
    'isoline.dash': [6, 2],
    'isoline.color': 'grey',
    'isoline.width': 0.75,
    
    'crossline.dash': [2, 2],
    'crossline.color': 'grey',
    'crossline.width': 0.5,
    'crossline.draw_limiting': True,
    
    'zeroline.dash': [2, 0],
    'zeroline.color': 'grey',
    'zeroline.width': 1,
    'zeroline.draw': True,
}


def _add_zeroline(ax):
    if cfg['zeroline.draw']:
        ax.axhline(0, color=cfg['zeroline.color'], lw=cfg['zeroline.width'], dashes=cfg['zeroline.dash'])
    return ax


def _validate_infer_indexes(pdf, x, y, indexes=None):
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

    if indexes is not None:
        if len(indexes) > 0:
            x = indexes[0]
        if len(indexes) > 1:
            y = indexes[1]

    if x is None and y is None:
        shape = pdf.shape
        if len(shape) == 2:
            lx, ly = shape

        elif len(shape) == 1:
            # pdf must be square-able
            s = int(shape[0] ** 0.5)
            if s ** 2 == shape[0]:
                pdf.shape = s, s
            else:
                raise ValueError('Cannot infer shape of flattened data array pdf since it does not have a square '
                                 'number of elements.')
            lx, ly = s, s

        # reconstruct x and y from shape
        x = np.linspace(0, 1, lx)
        y = np.linspace(0, 1, ly)

    elif (x is None and y is not None) or (x is not None and y is None):
        raise TypeError('x and y must be both set or both unset')

    elif x is not None and y is not None:
        if pdf.size != len(x) * len(y):
            raise ValueError('shapes of x,y and pdf do not match')
        pdf.shape = len(x), len(y)

    assert pdf.shape[0] == len(x) and pdf.shape[1] == len(y)
    return pdf, x, y


def add_level_lines(levels, ax, **kwargs):
    """Adds horizontal level lines to given axis."""

    for lvl in levels:
        ax.axhline(lvl, color=cfg['isoline.color'], lw=cfg['isoline.width'], dashes=cfg['isoline.dash'], **kwargs)


def add_cross_lines(levels, p, ax, **kwargs):
    """Adds vertical cross lines to given axis.
    See also utils.crossing_indexes.
    """
    crossings = utils.crossing_indexes(levels, p)
    for v in crossings:
        ax.axvline(v, color=cfg['crossline.color'], lw=cfg['crossline.width'], dashes=cfg['crossline.dash'], **kwargs)
    if cfg['crossline.draw_limiting']:
        ax.axvline(len(p), color=cfg['zeroline.color'], lw=cfg['zeroline.width'], dashes=cfg['zeroline.dash'])
    return ax


def density(levels, data, index=None, ax=None):
    """Plot data elements against its indexes and return axis object."""
    if ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(111)

    if index is None:
        index = np.arange(len(data))

    _add_zeroline(ax)
    add_level_lines(levels, ax)
    ax.plot(index, data)
    ax.set_ylim(bottom=0)

    return ax


def plot_sorted_density(levels, p, ax=None):
    """Plot sorted density `p` against index of sorted density value augmented with:
      * horizontal lines at each of the levels
      * vertical lines where the level lines cross the density plot
    """
    p = p.flatten()
    p_sorted = np.sort(p)

    if ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(111)

    _add_zeroline(ax)
    add_level_lines(levels, ax)
    add_cross_lines(levels, p_sorted, ax)
    ax.step(np.arange(p.size), p_sorted)
    ax.set_title("density")
    ax.set_ylim(bottom=0)
    return ax


def plot_cumulative_density(levels, p, ax=None):
    """Plot cumulative density of sorted density values `p`.
    The plot is augmented with vertical lines at the indexes of those p values are bounding the provided levels.
    """
    p = p.flatten()
    p_sorted = np.sort(p)
    p_cumsum = np.cumsum(p_sorted)

    level_cross_idx = utils.crossing_indexes(levels, p_sorted)

    if ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(111)

    _add_zeroline(ax)
    add_cross_lines(levels, p, ax)
    ax.step(np.arange(p.size), p_cumsum)
    ax.set_title("cumulative density")
    ax.set_ylim(bottom=0)
    return ax


def contour(p, x, y, levels=None, ax=None, **kwargs):
    """Draw contour plot over p, x, y.

    Optionally you may provide an axis to use for drawing and other arguments for `Axes.contour`.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if levels is not None:
        levels = np.unique(levels)
    ax.contour(x, y, p, levels=levels, **kwargs)
    return ax


def contour_levels_stat(levels, p, ax=None):
    """Plot contour level statistic for given levels and data."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    sum_prob = stats.contour_level_weight(levels, p)
    level_labels = ['L' + str(i + 1) for i in range(len(levels) + 1)]
    ax.bar(x=level_labels, height=sum_prob)
    return ax


def plot_contour_levels_stats(levels_lst, pdf_lst, labels=DEFAULT_LABELS):
    """Plot contour level statistics for all provided levels and data.

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
        contour_levels_stat(levels, pdf, ax[i])
        ax[i].set_title(labels[i])
    return ax


def contour_comparision_plot_2d(levels_lst, pdf, x=None, y=None, indexes=None, labels=DEFAULT_LABELS):
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

    pdf, x, y = _validate_infer_indexes(pdf, x, y, indexes)

    n = len(levels_lst)
    fig, ax = plt.subplots(figsize=(4 * n, 4), nrows=1, ncols=n)

    for i, (axis, label, levels) in enumerate(zip(ax, labels, levels_lst)):
        #print('levels ({}): {}'.format(label, levels))
        axis.contour(x, y, pdf, levels=np.unique(levels), colors=mpl.rcParams['lines.color'])
        axis.set_title(label)

    fig.show()
    return fig


def combined_2d(p, levels, x=None, y=None, indexes=None, slice_=None, ax=None):
    """Plot density contour plot over o using levels.

    Args:
        p : 2d-array-like
            The data to plot.
        levels : sequence of real numbers.
            Iso value levels.
        x : sequence of numbers, optional.
        y : sequence of numbers, optional.
            x and y together form the grid of input values where pdf provides function values.
            It must be either both, x and y be given, or none of them. If both are given it must match the shape of pdf
            or pdf must be linear, in which case the original shape of pdf is reconstructed from the lenght of x and y.
        indexes : 2d index sequence
            Is an alternative to x and y.
        slice_ : dict, optional.
            If given add a plot of a 1d slice of the 2d probability function. The slice is along axis `slice['axis']`
            at value `slice['value']` with density data `slice['pdf']`.
        figure : Figure object, optional.
            Use it for drawing, if provided.
    """

    figxsize = 5 if slice_ is not None else 4

    if ax is None:
        fig, ax = plt.subplots(1, figxsize, figsize=(4*figxsize, 4))
    else:
        assert(len(ax) >= figxsize)

    p, x, y = _validate_infer_indexes(p, x, y, indexes)

    axis_it = iter(ax)
    contour_ax = contour(p, x, y, levels, ax=next(axis_it))
    if slice_ is not None:
        # add slice indication line
        if slice_['axis'] == 'x':
            contour_ax.axvline(slice_['value'], color='red', lw=cfg['zeroline.width'], dashes=cfg['zeroline.dash'])
        else:
            contour_ax.axhline(slice_['value'], color='red', lw=cfg['zeroline.width'], dashes=cfg['zeroline.dash'])
        # draw slice
        density(levels, slice_['pdf'], x if slice_['axis'] == 'x' else y, ax=next(axis_it))
    plot_sorted_density(levels, p, ax=next(axis_it))
    plot_cumulative_density(levels, p, ax=next(axis_it))
    contour_levels_stat(levels, p, ax=next(axis_it))


def combined_1d(p, levels, index=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,4, figsize=(18, 4))
    else:
        assert(len(ax) >= 4), "require at least 4 axis objects"
    #p, x, y = _validate_infer_indexes(p, x=index)

    density(levels, p, index, ax=ax[0])
    plot_sorted_density(levels, p, ax=ax[1])
    plot_cumulative_density(levels, p, ax=ax[2])
    contour_levels_stat(levels, p, ax=ax[3])


def plot_combined(p, x=None, y=None, indexes=None, k=iso_levels.DEFAULT_K):
    """Plot both, contour level stats and the actual contour plots for different values of k"""

    p, x, y = _validate_infer_indexes(p, x, y, indexes)

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
        plot_contour_levels_stats(levels,
                                 len(levels)*[p],
                                 labels=[s.format(ki) for s in ['old (k={})', 'vertical (k={})', 'horizonal (k={})']])
    for i, ki in enumerate(k):
        contour_comparision_plot_2d([pdf_lvls[i], prob_lvls[i], prob_hori_lvls[i]],
                                    p,
                                    x, y,
                                    labels=[s.format(ki) for s in ['old (k={})', 'vertical (k={})', 'horizonal (k={})']])
