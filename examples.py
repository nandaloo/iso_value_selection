"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de

A collection of functions that generate a 2d support from probability density functions which partially are
derived from data.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal, gaussian_kde

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
    indexes = utils.normalize_to_indexes(low=[-2, -2], high=[2, 2], n=30)
    p = support_gaussian_2d(indexes=indexes)
    plotting.plot_combined(p, indexes[0], indexes[1], k=[3, 5, 7, 10])


def gaussian_2d_central_splike():
    indexes = utils.normalize_to_indexes(low=[-3, -3], high=[3, 3], n=50)
    p1 = support_gaussian_2d(indexes=indexes, sigma=[[1.0, 0], [0, 1]])
    p2 = support_gaussian_2d(indexes=indexes, sigma=[[0.005, 0], [0, 0.005]])
    plotting.plot_combined(p1+p2, indexes[0], indexes[1], k=[3, 5, 10])


def gaussian_2d_shifted_spike():
    indexes = utils.normalize_to_indexes(low=[-2, -2], high=[2, 2], n=50)
    p1 = support_gaussian_2d(indexes=indexes, sigma=[[1.0, 0], [0, 1]])
    p2 = support_gaussian_2d(indexes=indexes, mu=[0.25, 0.25], sigma=[[0.02, 0], [0, 0.02]])
    plotting.plot_combined(p1+p2, indexes[0], indexes[1], k=[3, 5, 10])


def gausssian_2d_three_gaussians():
    indexes = utils.normalize_to_indexes(low=[-2, -2], high=[2, 2], n=50)
    p1 = support_gaussian_2d(indexes=indexes, sigma=[[1, 0], [0, 1]])
    p2 = support_gaussian_2d(indexes=indexes, mu=[0.25, 0.25], sigma=[[0.02, 0], [0, 0.02]])
    p3 = support_gaussian_2d(indexes=indexes, mu=[-0.35, -0.35], sigma=[[0.07, 0], [0, 0.1]])
    plotting.plot_combined(2 * p1  + p2 + p3, indexes[0], indexes[1], k=[3, 5, 10])


def gaussian_1d():
    indexes = utils.normalize_to_indexes(low=[-2], high=[2], d=1, n=100)
    p = support_gaussian_1d(indexes=indexes)
    pass
    #  TODO


def allbus():
    df = pd.read_csv('allbus_age-vs-income.csv', index_col=False)
    allbus_p = df['p'].values

    plotting.plot_combined(allbus_p, k=[3, 5, 7, 10])


def titanic():
    df = pd.read_csv('titanic_age-vs-fare.csv', index_col=False)
    titanic_p = df['p'].values

    plotting.plot_combined(titanic_p, k=[3, 5, 7, 10])


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


def iris_kde(kernel_bandwidth=None):
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
    plotting.plot_combined(p, indexes[0], indexes[1], k=[3, 5, 10])


def basic_idea():
    import operator
    from matplotlib import pyplot as plt

    import iso_levels
    import stats

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

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    plotting.combined_1d(p_mixture_1d, levels2, index_1d, ax[0])
    plotting.combined_1d(p_mixture_1d, levels, index_1d, ax[1])
    fig.show()

    indexes_2d = utils.normalize_to_indexes(low=[-1, -3], high=[10, 3], n=100)
    p_single_2d = [support_gaussian_2d(indexes=indexes_2d, mu=[m, 0], sigma=[[s, 0], [0, s]]) for m, s in zip(mu, sigma)]
    p_mixture_2d = sum(map(operator.mul, weights, p_single_2d))
    levels = iso_levels.equi_prob_per_level(p_mixture_2d, k=7)
    levels2 = iso_levels.equi_value(p_mixture_2d, k=7)

    print('old embrace ratio: {}'.format(stats.embrace_ratio(levels2, p_mixture_2d)))
    print('new embrace ratio: {}'.format(stats.embrace_ratio(levels, p_mixture_2d)))

    # I used this to identify k=7 as particularly interesting
    #plotting.plot_combined(gp2d, indexes=gindex, k=list(range(2,10)))
    #plotting.plot_combined(gp2d, indexes=gindex, k=[7])

    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    plotting.combined_2d(p_mixture_2d, levels2, x=indexes_2d[0], y=indexes_2d[1], ax=ax[0])
    plotting.combined_2d(p_mixture_2d, levels, x=indexes_2d[0], y=indexes_2d[1], ax=ax[1])
    fig.show()


if __name__ == '__main__':

    iris_kde(0.15)
    gaussian_2d_plain()
    gaussian_2d_central_splike()
    gaussian_2d_shifted_spike()
    gausssian_2d_three_gaussians()
    allbus()
    titanic()

    # basic_idea()