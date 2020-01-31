"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de

A collection of functions that generate a 2d support from probability density functions which partially are
derived from data.
"""

import operator
import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal, gaussian_kde
from matplotlib import pyplot as plt

import iso_levels
import stats
import utils
import plotting


###### PDFs #######


def pdf_gauss_1d(x, mu=0.0, sigma=1):
    """Returns the density of the univariate normal distribution at x.

    Arsg:
        x : scalar or numpy-array like of scalars
            The point(s) to calculate the density for.
    """
    return norm.pdf(x, loc=mu, scale=sigma)


def pdf_gauss_2d(x, y):
    """Returns the density of the bivariate normal distribution at point (x,y)."""
    return pdf_gauss_1d(x)*pdf_gauss_1d(y)


def pdf_kernel(data, kernel_bandwidth=None):
    """Given data return a gaussian kernel density estimator.

    Args:
        data : the data as a numpy-like 2-d array where the first dimension indexes the attributes and the second
            indexes the data items
        kernel_bandwidth : see scipy.stats.gaussian_kde arguments

    Returns:
        A Kernel Density Estimator
    """
    return gaussian_kde(data, bw_method=kernel_bandwidth)


####### support generators ######


def support_gaussian_1d(indexes=None, mu=0.0, sigma=1):
    #return [pdf_gauss_1d(x) for x in indexes[0]]
    return pdf_gauss_1d(np.asarray(indexes), mu, sigma)


def support_gaussian_2d(indexes=None, mu=[0.0, 0.0], sigma=[[1.0, 0], [0, 1]]):
    """Generate support data for a specified multivariate gaussian and n steps along all dimensions.

    Returns:
        dict of x, y and pdf, where x and y are the grid axis support points and pdf the support points.
    """

    indexes = utils.normalize_to_indexes(indexes=indexes)
    input = utils.indexes_to_input_sequence(indexes)

    rnorm = multivariate_normal(mu, sigma)

    p = rnorm.pdf(input)
    p.shape = len(indexes[0]), len(indexes[1])
    return p


def support_mixed_gaussian_2d(indexes, mu, sigma, weights, from_scalar=True):
    """Return support value at indexes for specified mixture of gaussian.

    Args:
        indexes : 2d array
            the index arrays [x_index, y_index] where to get the support values
        mu : list of scalars, or list of list of 2-element scalars
            means of the mixture components
        sigma : list of scalars or list of 2d arrays of scalars
            standard deviation of mixture components
        weights : list of scalars
            the weight of each component
        from_scalar : bool, optional.
            If True mu and sigma must be list of scalars and are automatically extended required (larger) shape.
            Also see code.
    """
    if len(mu) != len(sigma) or len(sigma) != len(weights):
        raise ValueError('mu, sigma and weights do not have equal length')
    if from_scalar:
        mu = [[m, 0] for m in mu]
        sigma = [[[s, 0], [0, s]] for s in sigma]
    p_single_2d = [support_gaussian_2d(indexes=indexes, mu=m, sigma=s) for m, s in zip(mu, sigma)]
    return sum(map(operator.mul, weights, p_single_2d))


def support_kde(kernel, indexes=None):
    """Generate data for a kernel estimator `kernel`.

    Args:
        kernel : a kernel density estimator like scipy.stats.gaussian_kde arguments
        n : number of support points per axis
    """
    indexes = utils.normalize_to_indexes(indexes=indexes)
    input = utils.indexes_to_input_sequence(indexes)

    p = kernel.evaluate(input.transpose())
    p.shape = len(indexes[0]), len(indexes[1])
    return p


#### actual examples ####

def gaussian_2d_plain():
    lim = 3.5
    indexes = utils.normalize_to_indexes(low=[-lim, -lim], high=[lim, lim], n=100)
    p = support_gaussian_2d(indexes=indexes)
    plotting.plot_combined(p, indexes[0], indexes[1], k=[3, 5, 7, 10])


def gaussian_2d_central_splike():
    lim = 3.5
    indexes = utils.normalize_to_indexes(low=[-lim, -lim], high=[lim, lim], n=100)
    p1 = support_gaussian_2d(indexes=indexes, sigma=[[1.0, 0], [0, 1]])
    p2 = support_gaussian_2d(indexes=indexes, sigma=[[0.005, 0], [0, 0.005]])
    plotting.plot_combined(p1+p2, indexes[0], indexes[1], k=[3, 5, 10])


def gaussian_2d_shifted_spike():
    lim = 3
    indexes = utils.normalize_to_indexes(low=[-lim, -lim], high=[lim, lim], n=100)
    p1 = support_gaussian_2d(indexes=indexes, sigma=[[1.0, 0], [0, 1]])
    p2 = support_gaussian_2d(indexes=indexes, mu=[0.25, 0.25], sigma=[[0.02, 0], [0, 0.02]])
    plotting.plot_combined(p1+p2, indexes[0], indexes[1], k=[3, 5, 10])


def gausssian_2d_three_gaussians():
    lim = 2.5
    indexes = utils.normalize_to_indexes(low=[-lim, -lim], high=[lim, lim], n=100)
    p1 = support_gaussian_2d(indexes=indexes, sigma=[[1, 0], [0, 1]])
    p2 = support_gaussian_2d(indexes=indexes, mu=[0.25, 0.25], sigma=[[0.02, 0], [0, 0.02]])
    p3 = support_gaussian_2d(indexes=indexes, mu=[-0.35, -0.35], sigma=[[0.07, 0], [0, 0.1]])
    plotting.plot_combined(2 * p1 + p2 + p3, indexes[0], indexes[1], k=[3, 5, 10])


def gaussian_1d():
    indexes = utils.normalize_to_indexes(low=[-2], high=[2], d=1, n=100)
    p = support_gaussian_1d(indexes=indexes)
    pass
    #  TODO


def allbus():
    df = pd.read_csv('./data/allbus_age-vs-income.csv', index_col=False)
    allbus_p = df['p'].values
    plotting.plot_combined(allbus_p, k=[3, 5, 7, 10])


def titanic():
    df = pd.read_csv('./data/titanic_age-vs-fare.csv', index_col=False)
    titanic_p = df['p'].values
    plotting.plot_combined(titanic_p, k=[3, 5, 7, 10])


def data_file_with_kde(filepath, kernel_bandwidth=None, k=7, usecols=None, index_col=None):

    # get data
    df = pd.read_csv(filepath, index_col=index_col, usecols=usecols)
    data = df.values.transpose()

    # derive pdf
    mykernel = pdf_kernel(data, kernel_bandwidth=kernel_bandwidth)

    # get support
    indexes = utils.normalize_to_indexes(data=data)
    p = support_kde(mykernel, indexes)

    # plot
    #plotting.plot_combined(p, indexes[0], indexes[1], k=[3, 5, 7, 10])

    levels = iso_levels.equi_prob_per_level(p, k=k)
    levels2 = iso_levels.equi_value(p, k=k)

    # get index of max
    max_idx = np.unravel_index(np.argmax(p, axis=None), p.shape)
    slice_ = utils.get_slice(p, indexes, 'y', indexes[1][max_idx[0]])
    #slice_ = utils.get_slice(p, indexes, 'y', 68)

    # print('old embrace ratio: {}'.format(stats.embrace_ratio(levels2, p)))
    # print('new embrace ratio: {}'.format(stats.embrace_ratio(levels, p)))

    fig, ax = plt.subplots(2, 3, figsize=(3 * 5, 8))

    plotting.combined_2d(p, levels2, x=indexes[0], y=indexes[1], slice_=slice_, ax=ax[0])
    plotting.combined_2d(p, levels, x=indexes[0], y=indexes[1], slice_=slice_, ax=ax[1])
    fig.show()
    return fig


def titanic_kde(kernel_bandwidth=None):
    # get data
    titanic = pd.read_csv('./data/titanic_mixed.csv', index_col=None, usecols=['Age', 'Fare'])
    data = titanic.values.transpose()

    # derive pdf
    mykernel = pdf_kernel(data, kernel_bandwidth=kernel_bandwidth)

    # get support
    indexes = utils.normalize_to_indexes(data=data)
    p = support_kde(mykernel, indexes)

    # plot
    plotting.plot_combined(p, indexes[0], indexes[1], k=[3, 5, 7, 10])


def iris_kde(kernel_bandwidth=None, k=6):

    # get data
    from sklearn import datasets
    iris = datasets.load_iris()
    data = iris.data[:, :2].transpose()

    # derive pdf
    mykernel = pdf_kernel(data, kernel_bandwidth=kernel_bandwidth)

    # get support
    indexes = utils.normalize_to_indexes(data=data)
    p = support_kde(mykernel, indexes)

    # plot
    #plotting.plot_combined(p, indexes[0], indexes[1], k=[3, 5, 10])

    levels = iso_levels.equi_prob_per_level(p, k=k)
    levels2 = iso_levels.equi_value(p, k=k)

    slice_ = utils.get_slice(p, indexes, 'y', 3)
    #slice_ = None

    fig, ax = plt.subplots(2, 3, figsize=(3 * 5, 8))
    plotting.combined_2d(p, levels2, x=indexes[0], y=indexes[1], slice_=slice_, ax=ax[0])
    plotting.combined_2d(p, levels, x=indexes[0], y=indexes[1], slice_=slice_, ax=ax[1])
    fig.show()
    return fig


def broad_and_normal_gaussians(k=5):

    # works well!
    # mu = np.array([0, -5.5, 0, 5.5])
    # sigma = np.array([1, 10, 6, 10])
    # weights = np.array([1, 1, 0.5, 1])

    # more complex and still works well
    mu = np.array([
        [0, 0], [-5.5, -1], [0, 0], [5.5, 2], [-3, 4]
    ])
    sigma = np.array([1, 10, 6, 10, 9])
    sigma = [[[s, 0], [0, s]] for s in sigma]

    weights = np.array([1, 1, 0.5, 1, 1])

    indexes_2d = utils.normalize_to_indexes(low=[-12, -10], high=[12, 10], n=100)

    p = support_mixed_gaussian_2d(indexes_2d, mu, sigma, weights, from_scalar=False)

    levels = iso_levels.equi_prob_per_level(p, k=k)
    levels2 = iso_levels.equi_value(p, k=k)

    slice_ = utils.get_slice(p, indexes_2d, 'y', 0)

    fig, ax = plt.subplots(2, 3, figsize=(3 * 5, 8))
    plotting.combined_2d(p, levels2, x=indexes_2d[0], y=indexes_2d[1], slice_=slice_, ax=ax[0])
    plotting.combined_2d(p, levels, x=indexes_2d[0], y=indexes_2d[1], slice_=slice_, ax=ax[1])
    fig.show()
    return fig


def plateau(k=5):
    factor = 2.3

    mu = np.array([-0.5, 1, 3.5, 7, 6])-3
    sigma = np.array([0.05, 0.5, 0.2, 2, 0.3])*1.2
    weights = np.array([0.2, 1, 0.4, 1, 3])
    indexes_2d = utils.normalize_to_indexes(low=[-10, -10], high=[10, 10], n=100)

    # mu = [0]
    # sigma = [1]
    # weights = [1]
    # indexes_2d = utils.normalize_to_indexes(low=[-4, -4], high=[4, 4], n=100)

    p = support_mixed_gaussian_2d(indexes_2d, mu, sigma, weights)

    # raise p value
    p_raised = p + np.max(p)*factor

    # apply a circular 0--1 function
    for ix,x in enumerate(indexes_2d[0]):
        for iy,y in enumerate(indexes_2d[1]):
            if x*x + y*y > 32:
                p_raised[ix][iy] = 0

    # plot
    #plotting.plot_combined(p_raised, indexes=indexes_2d, k=[3,7,10])

    levels = iso_levels.equi_prob_per_level(p_raised, k=k)
    levels2 = iso_levels.equi_value(p_raised, k=k)

    slice_ = utils.get_slice(p_raised, indexes_2d, 'y', 0)

    fig, ax = plt.subplots(2, 3, figsize=(3 * 5, 8))
    plotting.combined_2d(p_raised, levels2, x=indexes_2d[0], y=indexes_2d[1], slice_=slice_, ax=ax[0])
    plotting.combined_2d(p_raised, levels, x=indexes_2d[0], y=indexes_2d[1], slice_=slice_, ax=ax[1])
    fig.show()
    return fig


def basic_idea():
    """Creates plot for initial explanatory and motivating example for paper.

    Also provide some search capabilities, i.e. allows to play with parameters to find exemplary distributions.
    """

    mu = [0, 1, 3.5, 7]
    sigma = [0.05, 0.5, 0.7, 2]
    weights = [0.6, 1, 0.7, 1]

    index_1d = utils.normalize_to_indexes(low=[-1], high=[10], n=2500, d=1)[0]
    p_single_1d = [support_gaussian_1d(index_1d, m, s) for m, s in zip(mu, sigma)]
    p_mixture_1d = sum(map(operator.mul, weights, p_single_1d))
    levels = iso_levels.equi_prob_per_level(p_mixture_1d, k=7)
    levels2 = iso_levels.equi_value(p_mixture_1d, k=7)

    # I used this i identify suitable mu, sigma and weights
    # figure = plt.figure(figsize=(9, 4))
    # ax = figure.add_subplot(121)
    # plotting.density(levels, gp1d, ax=ax)
    # ax = figure.add_subplot(122)
    # plotting.density(levels2, gp1d, ax=ax)

    fig1d, ax = plt.subplots(2, 3, figsize=(3*5, 8))
    plotting.combined_1d(p_mixture_1d, levels2, index_1d, ax[0])
    plotting.combined_1d(p_mixture_1d, levels, index_1d, ax[1])
    fig1d.show()

    indexes_2d = utils.normalize_to_indexes(low=[-1, -3], high=[10, 3], n=100)
    p_mixture_2d = support_mixed_gaussian_2d(indexes_2d, mu, sigma, weights)

    levels = iso_levels.equi_prob_per_level(p_mixture_2d, k=7)
    levels2 = iso_levels.equi_value(p_mixture_2d, k=7)

    slice_idx = int(len(indexes_2d[1])/2)
    slice_val = indexes_2d[1][slice_idx]
    slice_ = utils.get_slice(p_mixture_2d, indexes_2d, 'y', slice_val)

    print('old embrace ratio: {}'.format(stats.embrace_ratio(levels2, p_mixture_2d)))
    print('new embrace ratio: {}'.format(stats.embrace_ratio(levels, p_mixture_2d)))

    # I used this to identify k=7 as particularly interesting
    #plotting.plot_combined(gp2d, indexes=gindex, k=list(range(2,10)))
    #plotting.plot_combined(gp2d, indexes=gindex, k=[7])

    fig2d, ax = plt.subplots(2, 3, figsize=(3*5, 8))

    plotting.combined_2d(p_mixture_2d, levels2, x=indexes_2d[0], y=indexes_2d[1], slice_=slice_, ax=ax[0])
    plotting.combined_2d(p_mixture_2d, levels, x=indexes_2d[0], y=indexes_2d[1], slice_=slice_, ax=ax[1])
    fig2d.show()
    return fig1d, fig2d

if __name__ == '__main__':

    # iris_kde(0.15)
    #gaussian_2d_plain()
    #gaussian_2d_central_splike()
    #gaussian_2d_shifted_spike()
    #gausssian_2d_three_gaussians()
    #allbus()
    #titanic()
    #titanic_kde(0.15)

    #basic_idea()
    #for i,fig in enumerate(basic_idea()):
    #    fig.savefig("basic_idea_{}d.pdf".format(i+1))
    #plateau().savefig("plateau.pdf")
    #broad_and_normal_gaussians().savefig("broad_normal_gaussian.pdf")
    #data_file_with_kde('./data/nnc_z_min-linearity.csv', kernel_bandwidth=0.12).savefig("nnc_z_min-linearity.pdf")  # slice at maximum p

    #data_file_with_kde('~/Downloads/data.csv', kernel_bandwidth=0.10, k=6)

    iris_kde(0.15, k=5).savefig('iris_kde.pdf')
    # football works, but it's a bit to normal...
    # data_file_with_kde('./data/football_careergoals-vs-topspeed.csv', kernel_bandwidth=0.10, k=6)  # with slice at y = 60

    # data_file_with_kde('./data/mpg_year-mpg.csv', kernel_bandwidth=0.15)

    print("done")