from collections import namedtuple

import numpy as np
from numpy.random import random, normal, laplace, standard_t

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from simulations_framework import parallel_montecarlo, simulate_validation_vs_holdout_mse, average_and_std, pairs_average_and_std, save_figure
import pickler

from utils import Timer, Profiler, beep

RANDOM_SEED = 7

ParamsSparseLinearRegression = namedtuple('ParamsSparseLinearRegression', 'n_train n_validation n_holdout D df K_strong_columns strong_column_multiplier K')


def top_k_indices(a, k):
    assert 1 <= k <= len(a)
    return np.argpartition(a, -k)[-k:]


#class TopKVarianceVariableSelector(SelectorMixin, BaseEstimator):
#    def __init__(self, K):
#        self.K = K
#        self._is_fitted = False
#
#    def fit(self, X):
#        self._variances = np.var(X, axis=0)
#        self._selected_variables = top_k_indices(self._variances, self.K)
#        self._is_fitted = True
#        return self
#
#    def _get_support_mask(self):
#        assert self._is_fitted
#
#        mask = np.zeros(len(self._variances), np.bool)
#        mask[self._selected_variables] = True
#
#        return mask


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


class FirstKColumnsSelector:
    def __init__(self, K):
        self.K = K

    def fit(self, X):
        pass

    def transform(self, X):
        return X[:,:self.K]


class LastKColumnsSelector:
    def __init__(self, K):
        self.K = K

    def fit(self, X):
        pass

    def transform(self, X):
        return X[:,-self.K:]


class DatagenSparseDesignLinReg:
    def __init__(self, dimension, t_distribution_df, K_strong_columns, strong_column_multiplier):
        self.dimension = dimension
        self.t_distribution_df = t_distribution_df
        self.K_strong_columns = K_strong_columns
        self.strong_column_multiplier = strong_column_multiplier

        #self.feature_scales = 1 + strong_feature_multiplier*(random(size=(1,dimension)) < prob_strong_feature)
        self.beta = normal(size=(self.dimension,1))
        #print(f'Beta: {self.beta}')

    def generate(self, n):
        global X, Y
        X = standard_t(self.t_distribution_df, size=(n,self.dimension))
        #X = normal(size=(n,self.dimension))
        X[:,:self.K_strong_columns] *= self.strong_column_multiplier
        Y = (X @ self.beta).reshape(n)
        #print(f'X: {repr(X)}')
        #print(f'Y: {repr(Y)}')

        return (X,Y)


def simulate(params):
    return simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier),
            transformation = TopKVarianceVariableSelector(params.K),
            predictor = LinearRegression(fit_intercept=False))

def simulate_ss(params):
    return simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier),
            transformation = TopKSumSquares(params.K),
            predictor = LinearRegression(fit_intercept=False))

class CheatingLinearRegression:
    def __init__(self, datagen, transformation):
        self.datagen = datagen
        self.transformation = transformation

    def fit(self, X, Y):
        pass

    def predict(self, X):
        prediction = X @ self.datagen.beta[self.transformation._selected_variables]
        (n, ncol) = prediction.shape
        assert ncol == 1
        return prediction.reshape((n,))


def simulate_with_cheating(params):
    datagen = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier)
    transformation = TopKVarianceVariableSelector(params.K)
    return simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = datagen,
            transformation = transformation,
            predictor = CheatingLinearRegression(datagen, transformation))


def simulate_ss_with_cheating(params):
    datagen = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier)
    transformation = TopKSumSquares(params.K)
    return simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = datagen,
            transformation = transformation,
            predictor = CheatingLinearRegression(datagen, transformation))


def simulate_first_K_columns(params):
    return simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier),
            transformation = FirstKColumnsSelector(params.K),
            predictor = LinearRegression())


def simulate_last_K_columns(params):
    return simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier),
            transformation = LastKColumnsSelector(params.K),
            predictor = LinearRegression())


def print_results(list_of_validation_test_scores):
    n = len(list_of_validation_test_scores)
    (mean_val_error, mean_test_error, val_error_std, test_error_std) = pairs_average_and_std(list_of_validation_test_scores)
    print(f'Validation error: {mean_val_error:.3f}\u00B1{1.96*val_error_std/(n**0.5):.3f}')
    print(f'Test error: {mean_test_error:.3f}\u00B1{1.96*test_error_std/(n**0.5):.3f}')

def identity(x):
    return x

def test(k, simulation_function=simulate):
    print_results(parallel_montecarlo(None, simulation_function, identity, [ParamsSparseLinearRegression(n_train=10, n_validation=10, n_holdout=10, D=20, df=4, K_strong_columns=5, K=k)], n_repetitions=10000, seed=7)[0]) 


class ZeroPredictor:
    def fit(self, X, Y):
        pass

    def predict(self, X):
        (n,p) = X.shape
        return np.zeros(n)


class IdentityTransformation:
    def __init__(self, *args):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X


def simulate_null_model(params):
    gen = DatagenSparseDesignLinReg(params.D, params.df, params.K_strong_columns, params.strong_column_multiplier)
    (X,Y) = gen.generate(params.n_holdout)
    return np.mean(Y**2)


def precalc(simulation_function, n_range, D, df, K_strong_columns, strong_column_multiplier, K, reps):
    "Run multiple repetitions of simulate_gaussian_design() and save the results in a pickle file."
    fn_K2 = f'variable_selected_linear_regression_K2_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}'
    params_K2 = [ParamsSparseLinearRegression(n, n, n, D, df, K_strong_columns, strong_column_multiplier, K) for n in n_range]

    fn_LOO = f'variable_selected_linear_regression_LOO_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}'
    params_LOO = [ParamsSparseLinearRegression(n, 1, n, D, df, K_strong_columns, strong_column_multiplier, K) for n in n_range]

    for (filename, job_params) in [(fn_K2, params_K2), (fn_LOO, params_LOO)]:
        with Timer(f'{filename} ({reps} repetitions)'):
            parallel_montecarlo(filename, simulation_function, pairs_average_and_std, job_params, reps, seed=RANDOM_SEED)

    fn_null_model = f'variable_selected_linear_regression_null_model_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}'
    null_reps = reps*sum(n_range)
    with Timer(f'{fn_null_model} (repetitions {null_reps})'):
        parallel_montecarlo(fn_null_model, simulate_null_model, average_and_std, [ParamsSparseLinearRegression(0, 0, sum(n_range), D, df, K_strong_columns, strong_column_multiplier, -1)], reps, seed=RANDOM_SEED)
        


def plot_test_vs_validation_set(D, df, K_strong_columns, strong_column_multiplier, K, xlim=None, ylim=None, xticks=None, yticks=None):
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


    d = pickler.load(f'variable_selected_linear_regression_K2_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}')
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0-', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{val}}(m=n)$')
    plt.plot(x_values, d.results[:,1], 'C1-', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{gen}}(m=n)$')

    d = pickler.load(f'variable_selected_linear_regression_LOO_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}')
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0--', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{val}}(m=1)$')
    plt.plot(x_values, d.results[:,1], 'C1--', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{gen}}(m=1)$')

    d = pickler.load(f'variable_selected_linear_regression_null_model_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}')
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
    output_name = f'variable_selected_linear_regression_D{D}_df{df}_Kstrong{K_strong_columns}_multiplier{strong_column_multiplier}_K{K}_reps{d.n_repetitions}'
    save_figure(output_name)

def precalc_all(reps_lowdim, reps_highdim):
    print(f'Precalculating variable-selected linear regression D=100, t-distribution df=4, #repetitions = {reps_lowdim}')
    precalc(simulate, n_range=range(15,50,5), D=100, df=4, K_strong_columns=4, strong_column_multiplier=4, K=8, reps=reps_lowdim) 
    print(f'Precalculating variable-selected linear regression D=100, N(0,1), #repetitions = {reps_lowdim}')
    precalc(simulate, n_range=range(15,50,5), D=100, df=1000000, K_strong_columns=4, strong_column_multiplier=4, K=8, reps=reps_lowdim) 

    print(f'Precalculating variable-selected linear regression D=1000, t-distribution df=4, #repetitions = {reps_highdim}')
    precalc(simulate, n_range=range(20,160,10), D=1000, df=4, K_strong_columns=4, strong_column_multiplier=16, K=8, reps=reps_highdim) 
    print(f'Precalculating variable-selected linear regression D=1000, N(0,1), #repetitions = {reps_highdim}')
    precalc(simulate, n_range=range(20,160,10), D=1000, df=1000000, K_strong_columns=4, strong_column_multiplier=16, K=8, reps=reps_highdim) 

    print(f'Precalculating variable-selected linear regression D=50, t-distribution df=4, K=1, #repetitions = {reps_lowdim}')
    precalc(simulate, n_range=range(5,50,5), D=50, df=4, K_strong_columns=1, strong_column_multiplier=1, K=1, reps=reps_lowdim)
    print(f'Precalculating variable-selected linear regression D=50, N(0,1), K=1, #repetitions = {reps_lowdim}')
    precalc(simulate, n_range=range(5,50,5), D=50, df=1000000, K_strong_columns=1, strong_column_multiplier=1, K=1, reps=reps_lowdim)
    #precalc(simulate_ss, n_range=range(5,50,5), D=50, df=1000001, K_strong_columns=1, strong_column_multiplier=1, K=1, reps=reps_lowdim)

def plot_all():
    plot_test_vs_validation_set(D=100, df=4, K_strong_columns=4, strong_column_multiplier=4, K=8, xlim=[15,45], ylim=[200,800]) 
    plot_test_vs_validation_set(D=100, df=1000000, K_strong_columns=4, strong_column_multiplier=4, K=8)#, xlim=[15,45], ylim=[200,800]) 

    plot_test_vs_validation_set(D=1000, df=4, K_strong_columns=4, strong_column_multiplier=16, K=8, xlim=[20,150], ylim=[2000,6000], xticks=np.arange(20,160,20)) 
    plot_test_vs_validation_set(D=1000, df=1000000, K_strong_columns=4, strong_column_multiplier=16, K=8, xlim=[20,150], ylim=[1000,1500], xticks=np.arange(20,160,20))#, xlim=[20,200], ylim=[2000,6000]) 

    plot_test_vs_validation_set(D=50, df=4, K_strong_columns=1, strong_column_multiplier=1, K=1, xlim=[5,40], xticks=np.arange(5,45,5)) 
    plot_test_vs_validation_set(D=50, df=1000000, K_strong_columns=1, strong_column_multiplier=1, K=1, xlim=[5,40], xticks=np.arange(5,45,5)) 


