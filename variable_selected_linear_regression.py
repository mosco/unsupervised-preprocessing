"""
Supporting code for the paper
    "On the cross-validation bias due to unsupervised pre-processing" by Amit Moscovich and Saharon Rosset.
https://arxiv.org/abs/1901.08974v4

This module contains the simulation and plotting routines used to generate the main synthetic example of the paper:
Section "4. Main example: feature selection for high-dimensional linear regression". Figures 1 and 3.

Author:
    Amit Moscovich
    amit@moscovich.org
"""
from collections import namedtuple

import numpy as np
from numpy.random import random, normal, laplace, standard_t

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from simulations_framework import parallel_montecarlo, simulate_validation_vs_holdout_mse, average_and_std, pairs_average_and_std, save_figure, print_results
import pickler

from utils import Timer, Profiler, beep

RANDOM_SEED = 7
NOISE_MULTIPLIER = 1

ParamsSparseLinearRegression = namedtuple('ParamsSparseLinearRegression', 'n_train n_validation n_holdout D df K_strong_columns strong_column_multiplier K noise_variance')


def top_k_indices(a, k):
    assert 1 <= k <= len(a)
    return np.argpartition(a, -k)[-k:]


class TopKVarianceVariableSelector:
    def __init__(self, K):
        self.K = K
        self._is_fitted = False

    def fit(self, X):
        self._variances = np.var(X, axis=0)
        #print(f'Variances: {self._variances}')
        self._selected_variables = top_k_indices(self._variances, self.K)
        #print(f'Selected variables: {self._selected_variables}')
        self._mask = np.zeros(len(self._variances), np.bool)
        self._mask[self._selected_variables] = True
        self._is_fitted = True
        #assert set(self._selected_variables) == set(range(self.K))
        return self

    def transform(self, X):
        assert self._is_fitted

        return X[:,self._selected_variables]


class TopKSumSquares:
    def __init__(self, K):
        self.K = K
        self._is_fitted = False

    def fit(self, X):
        self._sumsquares = (X**2).sum(axis=0)
        #print(f'Variances: {self._variances}')
        self._selected_variables = top_k_indices(self._sumsquares, self.K)
        #print(f'Selected variables: {self._selected_variables}')
        self._mask = np.zeros(len(self._sumsquares), np.bool)
        self._mask[self._selected_variables] = True
        self._is_fitted = True
        #assert set(self._selected_variables) == set(range(self.K))
        return self

    def transform(self, X):
        assert self._is_fitted

        return X[:,self._selected_variables]


class DatagenSparseDesignLinReg:
    def __init__(self, dimension, t_distribution_df, K_strong_columns, strong_column_multiplier, noise_variance):
        self.dimension = dimension
        self.t_distribution_df = t_distribution_df
        self.K_strong_columns = K_strong_columns
        self.strong_column_multiplier = strong_column_multiplier
        assert noise_variance >= 0
        self.noise_variance = noise_variance

        #self.feature_scales = 1 + strong_feature_multiplier*(random(size=(1,dimension)) < prob_strong_feature)
        self.beta = normal(size=(self.dimension,1))
        #print(f'Beta: {self.beta}')

    def generate(self, n):
        global X, Y
        X = standard_t(self.t_distribution_df, size=(n,self.dimension))
        #X = normal(size=(n,self.dimension))
        X[:,:self.K_strong_columns] *= self.strong_column_multiplier
        Y = (X @ self.beta).reshape(n)
        noise = normal(scale=self.noise_variance**0.5, size=n)
        #print(f'X: {repr(X)}')
        #rint(f'Y: {repr(Y)}')
        #rint(f'noise: {repr(noise)}')

        return (X,Y+noise)


def simulate(params):
    return simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier, params.noise_variance),
            transformation = TopKVarianceVariableSelector(params.K),
            predictor = LinearRegression(fit_intercept=False))


def simulate_ss(params):
    return simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier, params.noise_variance),
            transformation = TopKSumSquares(params.K),
            predictor = LinearRegression(fit_intercept=False))


class ZeroPredictor:
    def fit(self, X, Y):
        pass

    def predict(self, X):
        (n,p) = X.shape
        return np.zeros(n)


def simulate_null_model(params):
    gen = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier, params.noise_variance)
    (X,Y) = gen.generate(params.n_holdout)
    return np.mean(Y**2)


def compute_noise_variance(D, K_strong_columns, strong_column_multiplier, noise_multiplier):
    """
    Recall that y = X * \beta
    where X is a vector of D independent samples from some distribution and \beta is a vector of N(0,1) variables.
    The first K_strong_columns of X are multiplied by strong_column_multiplier the variance of Y is proportional to
    K_strong_columns*(strong_column_multiplier**2) + (D-K_strong_columns)
    """
    return noise_multiplier*(K_strong_columns*(strong_column_multiplier**2) + (D-K_strong_columns))


def precalc(simulation_function, n_range, D, df, K_strong_columns, strong_column_multiplier, K, noise_multiplier, reps):
    "Run multiple repetitions of simulate_gaussian_design() and save the results in a pickle file."
    noise_variance = compute_noise_variance(D, K_strong_columns, strong_column_multiplier, noise_multiplier)
    fn_K2 = f'variable_selected_linear_regression_K2_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}_noisemul{noise_multiplier:.2f}'
    params_K2 = [ParamsSparseLinearRegression(n, n, n, D, df, K_strong_columns, strong_column_multiplier, K, noise_variance) for n in n_range]

    fn_LOO = f'variable_selected_linear_regression_LOO_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}_noisemul{noise_multiplier:.2f}'
    params_LOO = [ParamsSparseLinearRegression(n, 1, n, D, df, K_strong_columns, strong_column_multiplier, K, noise_variance) for n in n_range]

    for (filename, job_params) in [(fn_K2, params_K2), (fn_LOO, params_LOO)]:
        with Timer(f'{filename} ({reps} repetitions)'):
            parallel_montecarlo(filename, simulation_function, pairs_average_and_std, job_params, reps, seed=RANDOM_SEED)
        print('')

    fn_null_model = f'variable_selected_linear_regression_null_model_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_noisemul{noise_multiplier:.2f}'
    null_reps = reps*sum(n_range)
    with Timer(f'{fn_null_model} (repetitions {null_reps})'):
        parallel_montecarlo(fn_null_model, simulate_null_model, average_and_std, [ParamsSparseLinearRegression(0, 0, sum(n_range), D, df, K_strong_columns, strong_column_multiplier, -1, noise_variance)], reps, seed=RANDOM_SEED)
        

def plot_test_vs_validation_set(filename_prefix, D, df, K_strong_columns, strong_column_multiplier, K, noise_multiplier, xlim=None, ylim=None, xticks=None, yticks=None):
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

    d = pickler.load(f'variable_selected_linear_regression_K2_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}_noisemul{noise_multiplier:.2f}')
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0-', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{val}}(m=n)$')
    plt.plot(x_values, d.results[:,1], 'C1-', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{gen}}(m=n)$')

    d = pickler.load(f'variable_selected_linear_regression_LOO_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}_noisemul{noise_multiplier:.2f}')
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0--', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{val}}(m=1)$')
    plt.plot(x_values, d.results[:,1], 'C1--', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{gen}}(m=1)$')

    d = pickler.load(f'variable_selected_linear_regression_null_model_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_noisemul{noise_multiplier:.2f}')
    null_mse = d.results[0][0]
    plt.plot([min(x_values), max(x_values)], [null_mse, null_mse], 'k:', linewidth=1.0, label='Null model')

    plt.xlabel('$n$')
    plt.ylabel('MSE')
    plt.xlim(xlim if xlim is not None else [min(x_values), max(x_values)])
    if ylim is not None:
        plt.ylim(ylim)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    
    plt.legend(loc='best')
    output_name = f'{filename_prefix}_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}_noisemul{noise_multiplier:.2f}_reps{d.n_repetitions}'
    save_figure(output_name, f'noise_{noise_multiplier:.2f}')


def precalc_all(reps_lowdim=100000, reps_highdim=100000):
    # To simplify the code, we use t-distribution with 1,000,000 degrees of freedom when we want to generate Gaussian data.
    # (asymptotically the t distribution converges to N(0,1))

    print('='*80)
    print(f'Precalculating variable-selected linear regression D=100, t-distribution df=4, #repetitions = {reps_lowdim}')
    precalc(simulate, n_range=range(20,65,5), D=100, df=4, K_strong_columns=5, strong_column_multiplier=5, K=10, noise_multiplier=NOISE_MULTIPLIER, reps=reps_lowdim) 

    print('='*80)
    print(f'Precalculating variable-selected linear regression D=100, N(0,1), #repetitions = {reps_lowdim}')
    precalc(simulate, n_range=range(20,65,5), D=100, df=1000000, K_strong_columns=5, strong_column_multiplier=5, K=10, noise_multiplier=NOISE_MULTIPLIER, reps=reps_lowdim) 

    print('='*80)
    print(f'Precalculating variable-selected linear regression D=1000, t-distribution df=4, #repetitions = {reps_highdim}')
    precalc(simulate, n_range=range(200,650,50), D=1000, df=4, K_strong_columns=10, strong_column_multiplier=10, K=100, noise_multiplier=NOISE_MULTIPLIER, reps=reps_highdim) 

    print('='*80)
    print(f'Precalculating variable-selected linear regression D=1000, N(0,1), #repetitions = {reps_highdim}')
    precalc(simulate, n_range=range(200,650,50), D=1000, df=1000000, K_strong_columns=10, strong_column_multiplier=10, K=100, noise_multiplier=NOISE_MULTIPLIER, reps=reps_highdim) 

    print('='*80)
    print(f'Precalculating variable-selected linear regression D=50, t-distribution df=4, K=1, #repetitions = {reps_lowdim}')
    precalc(simulate_ss, n_range=range(5,45,5), D=50, df=4, K_strong_columns=1, strong_column_multiplier=1, K=1, noise_multiplier=0, reps=reps_lowdim)

    print('='*80)
    print(f'Precalculating variable-selected linear regression D=50, N(0,1), K=1, #repetitions = {reps_lowdim}')
    precalc(simulate_ss, n_range=range(5,45,5), D=50, df=1000000, K_strong_columns=1, strong_column_multiplier=1, K=1, noise_multiplier=0, reps=reps_lowdim)


def plot_all():
    plot_test_vs_validation_set('variance_filtered_linear_regression', D=100, df=4, K_strong_columns=5, strong_column_multiplier=5, K=10, noise_multiplier=NOISE_MULTIPLIER, xticks=range(20,65,5))#, xlim=[20,60], ylim=[200,400]) 
    plot_test_vs_validation_set('variance_filtered_linear_regression',D=100, df=1000000, K_strong_columns=5, strong_column_multiplier=5, K=10, noise_multiplier=NOISE_MULTIPLIER,  xticks=range(20,65,5))#, xlim=[20,60], ylim=[100,180])

    plot_test_vs_validation_set('variance_filtered_linear_regression', D=1000, df=4, K_strong_columns=10, strong_column_multiplier=10, K=100, noise_multiplier=NOISE_MULTIPLIER,  xticks=range(200,650,50))#, xlim=[200,600], ylim=[2000,4000])
    plot_test_vs_validation_set('variance_filtered_linear_regression', D=1000, df=1000000, K_strong_columns=10, strong_column_multiplier=10, K=100, noise_multiplier=NOISE_MULTIPLIER,  xticks=range(200,650,50))#, xlim=[200,600], ylim=[1000,1800])

    plot_test_vs_validation_set('norm_filtered_linear_regression', D=50, df=4, K_strong_columns=1, strong_column_multiplier=1, K=1, noise_multiplier=0, xticks=np.arange(5,45,5)) 
    plot_test_vs_validation_set('norm_filtered_linear_regression', D=50, df=1000000, K_strong_columns=1, strong_column_multiplier=1, K=1, noise_multiplier=0, xticks=np.arange(5,45,5)) 


