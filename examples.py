"""
Copyright 2018 Philipp Lucas, philipp.lucas@dlr.de

A collection of functions that generate a 2d support from probability density functions which partially are
derived from data.
"""

import pandas as pd
from scipy.stats import norm, multivariate_normal, gaussian_kde

import utils
import plotting


###### PDFs #######


def pdf_gauss_1d(x):
    """Returns the density of the univariate normal distribution at point x."""
    return norm.pdf(x)


def pdf_gauss_2d(x, y):
    """Returns the density of the bivariate normal distribution at point (x,y)."""
    return pdf_gauss_1d(x)*pdf_gauss_1d(y)


def pdf_kernel(data, kernel_bandwidth=None):
    """Given data return a gaussian kernel density estimator.

    Args:
        data : the data
        kernel_bandwidth : see scipy.stats.gaussian_kde arguments

    Returns:
        A Kernel Density Estimator
    """
    return gaussian_kde(data, bw_method=kernel_bandwidth)


####### support generators ######


def support_gaussian_1d(indexes=None):
    return [pdf_gauss_1d(x) for x in indexes[0]]


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
    plotting.plot_combined(p1, indexes[0], indexes[1], k=[3, 5, 10])


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
    plotting.plot_combined( 2 * p1  + p2 + p3, indexes[0], indexes[1], k=[3, 5, 10])


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


if __name__ == '__main__':

    iris_kde(0.15)
    # gaussian_2d_plain()
    # gaussian_2d_central_splike()
    # gaussian_2d_shifted_spike()
    # gausssian_2d_three_gaussians()
    # allbus()
    # titanic()
