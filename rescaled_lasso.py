"""
Supporting code for the paper
    "Rescaling and other forms of unsupervised preprocessing may bias cross-validation"
    by Amit Moscovich and Saharon Rosset.

This module contains the simulation and graph plotting routines used for making the rescaled Lasso
figures, which appear in section "Example 2: rescaling prior to Lasso linear regression".

Author:
    Amit Moscovich
    amit@moscovich.org
"""

from collections import Counter, namedtuple

from numpy import sum, arange, dot, newaxis, sqrt, diagonal, absolute, sign
from mkl_random import normal, choice
from numpy.linalg import svd, norm, qr
from scipy.stats import ortho_group
from sklearn.linear_model import Lasso

from simulations_framework import parallel_montecarlo, simulate_validation_vs_holdout_mse, pairs_average_and_std, save_figure
from utils import Timer
import pickler


RANDOM_SEED = 7


# This namedtuple holds the parameters for a single simulation of the rescaled Lasso.
# n_train - number of training samples.
# n_validation - number of validation samples. e.g. in 2-fold cross-validation n_validation is equal to n_train.
# n_holdout - number of test, or hold-out samples, used to estimate the generalization error.
#             Unlike validation samples, these samples are not involved in forming the unsupervised transformation.
# D - the dimension of the feature vectors
# sigma - noise level
# alpha - Lasso regularization parameter
ParamsRescaledLasso = namedtuple('ParamsRescaledLasso', 'n_train n_validation n_holdout D sigma alpha')


class ScalerAssumingZeroMean:
    """
    This preliminary unsupervised transformation is similar to sklearn.preprocessing.StandardScaler,
    which rescales every feature to have zero mean and unit variance.
    The only difference is that this transformer assumes that the mean is zero, rather than estimating it.
    """
    def fit(self, X):
        (n,p) = X.shape

        self.feature_std = sqrt(sum(X**2,0)/n)

    def transform(self, X):
        return X / self.feature_std[newaxis,:]


class DatagenGaussianDesignDotprod:
    """
    This class generates data according to the sampling distribution of Example 2
    in the paper (rescaling prior to Lasso linear regression)
    """ 
    def __init__(self, dimension, sigma):
        self.dimension = dimension
        self.sigma = sigma
        self.v = normal(size=(self.dimension,))

    def generate(self, n):
        X = normal(size=(n,self.dimension)) / (self.dimension**0.5)
        Y = dot(X, self.v) 
        if self.sigma != 0:
            Y += normal(scale=self.sigma, size=n)
        return (X,Y)


def simulate_gaussian_design(params):
    "A single simulation run of Example 2 from the paper"
    return simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = DatagenGaussianDesignDotprod(params.D, params.sigma),
            transformation = ScalerAssumingZeroMean(),
            predictor = Lasso(params.alpha, fit_intercept=False, tol=0.01))


def run_precalc_gaussian_design(filename, sizes, D, sigma, alpha, n_repetitions):
    "Run multiple repetitions of simulate_gaussian_design() and save the results in a pickle file."
    job_params = [ParamsRescaledLasso(n_train, n_validation, n_holdout, D, sigma, alpha)
        for (n_train, n_validation, n_holdout) in sizes]
    parallel_montecarlo(filename, simulate_gaussian_design, pairs_average_and_std, job_params, n_repetitions, seed=RANDOM_SEED)


def precalc(D, sigma, alpha, n_range, reps):
    "Run figure precalculations for both 2-fold cross-validation and leave-one-out cross-validation."

    print(f'n_range: {n_range}')

    filename = f'rescaledlasso_gaussiandesign_K2_D{D}_sigma{sigma:.2f}_alpha{alpha:.2f}'.replace('.','_')
    with Timer(f'{filename} ({reps} repetitions)'):
        run_precalc_gaussian_design(filename, [(n,n,n) for n in n_range], D, sigma, alpha, reps)

    filename = f'rescaledlasso_gaussiandesign_LOO_D{D}_sigma{sigma:.2f}_alpha{alpha:.2f}'.replace('.','_')
    with Timer(f'{filename} ({reps} repetitions)'):
        run_precalc_gaussian_design(filename, [(n,1,n) for n in n_range], D, sigma, alpha, reps)


def precalc_all(lowdim_reps, highdim_reps):
    """
    Precalculate results needed for all the rescaled Lasso figures in the paper.

    This takes aroun 1.5 years of computation on a single core using the parameters from the paper (lowdim_reps=10**7 highdim_reps=10**6)
    so run this on a multicore machine or use less repetitions.
    """
    for sigma in [0, 0.1, 1]:
        print('Precalculating low-dimensional rescaled Lasso, sigma =', sigma, ' #repetitions = ', lowdim_reps)
        print('Each of these simulations in dimension D=5 takes around 10 hours per 1 million iterations on a 2016 Intel Xeon core')
        precalc(D=5, sigma=sigma, alpha=0.5, n_range=arange(10,110,10), reps=lowdim_reps)

    for sigma in [0, 1, 10]:
        print('Precalculating noiseless high-dimensional rescaled Lasso, sigma =', sigma, ' #repetitions = ', highdim_reps)
        print('Each of these simulations in dimension D=10,000 takes around 180 days per 1 million iterations on a single 2016 Intel Xeon core.')
        precalc(D=10000, sigma=sigma, alpha=0.1, n_range=range(20,220,20), reps=highdim_reps)


def plot_test_vs_validation_set(D, sigma, alpha, xlim=None, ylim=None, xticks=None, yticks=None):
    """
    Plot a single figure which compares the expected validation and generalization errors
    for various numbers of training samples (n), using either m=1 or m=n validation samples.
    """
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.style.use('./latex-paper.mplstyle')
    plt.figure()
    ax = plt.axes()
    ax.yaxis.grid(True)

    d = pickler.load(f'rescaledlasso_gaussiandesign_K2_D{D}_sigma{sigma:.2f}_alpha{alpha:.2f}'.replace('.','_'))
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0-', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{val}}(m=n)$')
    plt.plot(x_values, d.results[:,1], 'C1-', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{gen}}(m=n)$')

    d = pickler.load(f'rescaledlasso_gaussiandesign_LOO_D{D}_sigma{sigma:.2f}_alpha{alpha:.2f}'.replace('.','_'))
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0--', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{val}}(m=1)$')
    plt.plot(x_values, d.results[:,1], 'C1--', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{gen}}(m=1)$')

    plt.xlabel('n')
    plt.ylabel('MSE')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    
    plt.legend(loc='best')
    output_name = f'rescaled_lasso_D{D}_sigma{sigma:.2f}_alpha{alpha:.2f}_reps{d.n_repetitions}'.replace('.','_')
    save_figure(output_name)


def plot_all():
    plot_test_vs_validation_set(D=5, sigma=0, alpha=0.5, xticks=arange(10,110,10), xlim=[10,100], ylim=[0.59,0.68])
    plot_test_vs_validation_set(D=5, sigma=0.1, alpha=0.5, xticks=arange(10,110,10), xlim=[10,100], ylim=[0.6, 0.69])
    plot_test_vs_validation_set(D=5, sigma=1.0, alpha=0.5, xticks=arange(10,110,10), xlim=[10,100], ylim=[1.58,1.67])
    plot_test_vs_validation_set(D=10000, sigma=0, alpha=0.1, xticks=arange(20,220,20), xlim=[20,200], yticks=arange(1.07,1.18,0.01), ylim=[1.07,1.17])
    plot_test_vs_validation_set(D=10000, sigma=1.0, alpha=0.1, xticks=arange(20,220,20), xlim=[20,200], yticks=arange(2.2,2.36,0.02), ylim=[2.2,2.36])
    plot_test_vs_validation_set(D=10000, sigma=10.0, alpha=0.1, xticks=arange(20,220,20), xlim=[20,200])


