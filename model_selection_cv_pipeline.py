"""
Supporting code for the paper
    "On the cross-validation bias due to unsupervised pre-processing" by Amit Moscovich and Saharon Rosset.
https://arxiv.org/abs/1901.08974v4

This module contains the simulation and graph plotting routines used for running the cross-validation-based
model selection experiments that appear in Section "6. Potential impact on model selection".

Author:
    Amit Moscovich
    amit@moscovich.org
"""
from collections import namedtuple

import numpy as np
from numpy.random import normal, standard_t

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, check_array
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error as mse

import rescaled_lasso
from simulations_framework import parallel_montecarlo, mean_confidence_interval, save_figure
from utils import Timer
import pickler

import matplotlib.pyplot as plt

RANDOM_SEED = 7331


def _check_sizes(X0, Y0, X1, Y1):
    (n0,p0) = X0.shape
    assert Y0.shape == (n0,)

    (n1,p1) = X1.shape
    assert Y1.shape == (n1,)

    assert p0 == p1


def correct_pipeline_mse(X, Y, X_holdout, Y_holdout, preprocessing_transformer, predictor, predictor_param_grid, num_cv_folds):
    _check_sizes(X, Y, X_holdout, Y_holdout)

    pipeline = Pipeline([('preprocessor', preprocessing_transformer), ('predictor', predictor)])

    param_grid = {f'predictor__{name}': val for (name, val) in predictor_param_grid.items()}
    cv = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', n_jobs=1, refit=True, cv=num_cv_folds)
    cv.fit(X, Y)

    Y_holdout_pred = cv.predict(X_holdout)
    prediction_mse = mse(Y_holdout_pred, Y_holdout)

    assert len(cv.best_params_) == 1
    best_param = list(cv.best_params_.values())[0]

    return (prediction_mse, best_param)


def incorrect_pipeline_mse(X, Y, X_holdout, Y_holdout, preprocessing_transformer, predictor, predictor_param_grid, num_cv_folds):
    _check_sizes(X, Y, X_holdout, Y_holdout)
    #assert (X.shape[0] % num_cv_folds)

    X_preprocessed = preprocessing_transformer.fit_transform(X)
    cv = GridSearchCV(predictor, predictor_param_grid, scoring='neg_mean_squared_error', n_jobs=1, refit=True, cv=num_cv_folds)
    cv.fit(X_preprocessed, Y)
    
    Y_holdout_pred = cv.predict(preprocessing_transformer.transform(X_holdout))
    prediction_mse = mse(Y_holdout_pred, Y_holdout)

    assert len(cv.best_params_) == 1
    best_param = list(cv.best_params_.values())[0]

    return (prediction_mse, best_param)


def compare_cv_mse(datagen, n_train_val, n_holdout, preprocessing_transformer, predictor, predictor_param_grid, num_cv_folds):
    (X,Y) = datagen.generate(n_train_val)
    (X_holdout, Y_holdout) = datagen.generate(n_holdout)

    (correct_mse, correct_best_param)  = correct_pipeline_mse(X, Y, X_holdout, Y_holdout, preprocessing_transformer, predictor, predictor_param_grid, num_cv_folds)
    (incorrect_mse, incorrect_best_param) = incorrect_pipeline_mse(X, Y, X_holdout, Y_holdout, preprocessing_transformer, predictor, predictor_param_grid, num_cv_folds)

    null_model_mse = np.mean([y**2 for y in Y_holdout])

    return (correct_mse, correct_best_param, incorrect_mse, incorrect_best_param, null_model_mse)


ParamsRescaledLassoPipeline = namedtuple('ParamsRescaledLassoPipeline', 'n_train n_validation n_holdout p df sigma with_intercept lasso_alpha_range')

class DatagenTDesignDotprod:
    def __init__(self, dimension, df, sigma, with_intercept):
        self.dimension = dimension
        self.df = df
        self.sigma = sigma
        #self.v = normal(size=(self.dimension,))
        self.beta = standard_t(df, size=(self.dimension,))
        self.intercept = standard_t(df, 1)[0] if with_intercept else 0.0


    def generate(self, n):
        #X = normal(size=(n,self.dimension))
        #X = standard_t(self.df, size=(n,self.dimension))
        X = standard_t(self.df, size=(self.dimension,n)).transpose() # By transposing the array, we get Fortran memory ordering which accelerates algorithms that work feature-by-feature like the Lasso fit.
        Y = np.dot(X, self.beta) + self.intercept
        if self.sigma != 0:
            Y += normal(scale=self.sigma, size=n)
        return (X,Y)

def compare_normalized_lasso_cv_mse(params):
    n = params.n_train
    m = params.n_validation
    assert ((n+m) % m) == 0
    num_cv_folds = int((n+m)/m)
    assert type(params.with_intercept) == bool

    #datagen = rescaled_lasso.DatagenGaussianDesignDotprod(params.p, params.sigma)
    datagen = DatagenTDesignDotprod(params.p, params.df, params.sigma, params.with_intercept)
    #datagen = DatagenTDesignDotProd(params.p, params.sigma, 4)
    preprocessing_transformer = rescaled_lasso.ScalerAssumingZeroMean()
    predictor = Lasso(fit_intercept=params.with_intercept, tol=0.001)
    return compare_cv_mse(datagen, n+m, params.n_holdout, preprocessing_transformer, predictor, {'alpha': params.lasso_alpha_range}, num_cv_folds)


def precalc(n_range, p, df, sigma, with_intercept, lasso_alpha_range, reps):
    #params = [ParamsRescaledLassoPipeline(n-1, 1, 10*n, p, sigma, lasso_alpha_range) for n in n_range]
    for n in n_range:
        assert (n%10) == 0
    #params = [ParamsRescaledLassoPipeline(n_train=int(n*4/5), n_validation=int(n/5), n_holdout=100*n, p=p, df=df, sigma=sigma, lasso_alpha_range=lasso_alpha_range) for n in n_range]
    params = [ParamsRescaledLassoPipeline(n_train=int(n*9/10), n_validation=int(n/10), n_holdout=100*n, p=p, df=df, sigma=sigma, with_intercept=with_intercept, lasso_alpha_range=lasso_alpha_range) for n in n_range]
    filename = f'normalized_lasso_cv_pipeline_p{p}_df{df}_sigma{sigma}_intercept{with_intercept}_10FOLDCV'
    with Timer(f'{filename} ({reps} repetitions)'):
        parallel_montecarlo(filename, compare_normalized_lasso_cv_mse, lambda x:x, params, reps, seed=RANDOM_SEED)


def read_results_with_confintervals(p, df, sigma, with_intercept):
    d = pickler.load(f'normalized_lasso_cv_pipeline_p{p}_df{df}_sigma{sigma}_intercept{with_intercept}_10FOLDCV')
    assert d.results.shape == (len(d.xs), d.n_repetitions, 5)

    null_model_mse = np.mean(d.results[:,:,4])

    correct_mses = np.apply_along_axis(mean_confidence_interval, axis=1, arr=d.results[:,:,0])
    assert correct_mses.shape == (len(d.xs), 2)

    incorrect_mses = np.apply_along_axis(mean_confidence_interval, axis=1, arr=d.results[:,:,2])
    assert incorrect_mses.shape == (len(d.xs), 2)

    n_range = d.xs

    return (d.xs, null_model_mse, correct_mses, incorrect_mses, d.n_repetitions)


def print_res(p, df, sigma, with_intercept):
    (xs, null_model_mse, correct_mses, incorrect_mses, n_reps) = read_results_with_confintervals(p, df, sigma, with_intercept)

    print('Null model MSE:')
    print(null_model_mse)
    print()

    print('Correct MSEs:')
    print(correct_mses)
    print()

    print('Incorrect MSEs:')
    print(incorrect_mses)


def plot_res(p, df, sigma, with_intercept):
    (xs, null_model_mse, correct_mses, incorrect_mses, n_reps) = read_results_with_confintervals(p, df, sigma, with_intercept)


    plt.style.use('./latex-paper.mplstyle')
    fig, ax = plt.subplots()
    X = [x.n_train+x.n_validation for x in xs]
    ax.fill_between(X, correct_mses[:,0], correct_mses[:,1], alpha=0.2)
    ax.fill_between(X, incorrect_mses[:,0], incorrect_mses[:,1], alpha=0.2)
    ax.plot(X, np.mean(correct_mses, axis=1), '-', label='Correct')
    ax.plot(X, np.mean(incorrect_mses, axis=1), '-', label='Incorrect')
    ax.plot([X[0], X[-1]], [null_model_mse]*2, 'k:', linewidth=1.0, label='Null model')
    ax.legend()
    ax.set_xticks(X)
    ax.set_xlim([X[0], X[-1]])
    ax.set_xlabel('N')
    ax.set_ylabel('MSE')
    #latexsigma = r'\sigma'
    #ax.set_title(f'$p={p} \ df={df} \ {latexsigma}={sigma} \ intr={int(with_intercept)} \ reps={n_reps}$')
    ax.grid(axis='y')

    save_figure(f'normalized_lasso_cv_pipeline_p{p}_df{df}_sigma{sigma}_10FOLDCV_reps{n_reps}', 'normalized_lasso_pipeline')
    

def plot_alphas_correct_vs_incorrect(p, df, sigma, with_intercept, i, bins='auto'):
    plt.figure()
    results = pickler.load(f'normalized_lasso_cv_pipeline_p{p}_df{df}_sigma{sigma}_intercept{with_intercept}_10FOLDCV').results[i]
    assert results.ndim == 2
    correct_alphas = results[:,1]
    incorrect_alphas = results[:,3]
    plt.hist([correct_alphas, incorrect_alphas], label=['correct', 'incorrect'], bins=bins)
    plt.legend()


def precalc_all(reps_low_dim, reps_high_dim):
    precalc(range(20,60,10), p=100, df=4, sigma=10.0, with_intercept=False, lasso_alpha_range=[(2**i) for i in range(-10, 11)], reps=reps_low_dim)
    precalc(range(100,300,50), p=1000, df=4, sigma=40.0, with_intercept=False, lasso_alpha_range=[(2**i) for i in range(-10, 11)], reps=reps_high_dim)


def plot_all():
    plot_res(p=100, df=4, sigma=10.0, with_intercept=False)
    plot_res(p=1000, df=4, sigma=40.0, with_intercept=False)

