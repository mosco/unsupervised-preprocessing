"""
Supporting code for the paper
    "Rescaling and other forms of unsupervised preprocessing may bias cross-validation"
    by Amit Moscovich and Saharon Rosset.

This simulations framework contains code that is used by both the Example 1 in the paper (grouping_rare_categories module)
and by Example 2 (rescaled_lasso module)

Author:
    Amit Moscovich
    amit@moscovich.org
"""
import os
import functools
import itertools
import multiprocessing
import socket
import itertools


import numpy as np
import scipy
import mkl
import mkl_random
from sklearn.utils import shuffle

from utils import Timer
import pickler


FIGURES_PATH = 'figures/'
DPI = 100


def get_n_cpu():
    hostname = socket.gethostname()
    if hostname == 'latte.math.princeton.edu.private':
        return 40
    elif hostname == 'chai.math.princeton.edu.private':
        return 60
    else:
        return max(1, int(multiprocessing.cpu_count()/2))


def set_random_seed_and_apply_func(func, random_seed, func_params):
    # We seed both random number generators, since func may use either or both.
    np.random.seed(random_seed)
    mkl_random.seed(random_seed)
    return func(func_params)


def parallel_montecarlo(filename, mapper, reducer, jobs_params, n_repetitions, seed=None, n_cpu = None):
    """
    This function implements a basic map-reduce framework based on multiprocessing.Pool.

    Inputs:
        filename - name of output, where the results will be saved as a pickle.
        mapper - this is the function that runs the computaitonal job, given an element of job_params.
        reducer - function for aggregating n_repetitions runs of a job.
        job_params - list of job parameters.
        n_repetitions - number of times to run each job.
        seed - Random seed to be used. To have reproducible results, always specify a seed.
        n_cpu - number of processes to use. The default is to use all available cores.

    Outputs:
        reduced_results - output of reducer on the various simulations
        Also, the results will be saved to a pickle file

    Example: (this computes the means of 3 normal random variables with different means)
    
    >> parallel_montecarlo('testing', numpy.random.normal, numpy.mean, [-1,0,+1], 1000)
        n_cpu: 4
        Saving to ./pickles/testing.pickle.gz
        Saved fields:  n_repetitions, name, results, seed, xs
        Out[14]: [-0.9465148770830919, 0.03763575004851667, 1.056358627427924]
    """
    mkl.set_num_threads(1)
    if n_cpu is None:
        n_cpu = get_n_cpu()
    #print(f'n_cpu: {n_cpu}')

    SEED = seed if seed is not None else 0
    N_SEED_INTS = 4
    mkl_random.seed(SEED)
    iteration_parameters = zip(mkl_random.randint(0, 2**32, size=(len(jobs_params)*n_repetitions, N_SEED_INTS)), itertools.cycle(jobs_params)) 

    wrapped_job_computation_func = functools.partial(set_random_seed_and_apply_func, mapper)
    if n_cpu == 1:
        results = list(itertools.starmap(wrapped_job_computation_func, iteration_parameters))
    else:
        with multiprocessing.Pool(processes=n_cpu) as p:
            results = list(p.starmap(wrapped_job_computation_func, iteration_parameters))

    results_grouped_by_params = [results[i::len(jobs_params)] for i in range(len(jobs_params))]
    reduced_results = list(map(reducer, results_grouped_by_params))

    if filename is not None:
        pickler.dump(filename, name=filename, xs=jobs_params, results=np.array(reduced_results), n_repetitions=n_repetitions, seed=SEED)

    return reduced_results


def mean_squared_error(Y_true, Y_predicted):
    return np.mean((Y_true - Y_predicted)**2)


def simulate_validation_vs_holdout_mse(n_train, n_validation, n_holdout, data_generator, transformation, predictor):
    """
    This is the main function used to estimate the validation and generalization errors for a predictor
    which is run on samples that underwent a preliminary unsupervised transformation.
    """
    #(X_unshuffled,Y_unshuffled) = data_generator.generate(n_train+n_validation+n_holdout)
    #(X,Y) = shuffle(X_unshuffled, Y_unshuffled)
    (X,Y) = data_generator.generate(n_train+n_validation+n_holdout)

    Xtrain = X[:n_train]
    Ytrain = Y[:n_train]
    Xvalidation = X[n_train:n_train+n_validation]
    Yvalidation = Y[n_train:n_train+n_validation]
    Xholdout = X[n_train+n_validation:]
    Yholdout = Y[n_train+n_validation:]

    Xtrainval = X[:n_train+n_validation]
    transformation.fit(Xtrainval)
    #print(f'Transformed X: {transformation.transform(X)}')
    predictor.fit(transformation.transform(Xtrain), Ytrain)
    #print(f'predictor.coef_: {predictor.coef_}')

    Yvalidation_pred = predictor.predict(transformation.transform(Xvalidation))
    #print(f'Yvalidation_pred = {Yvalidation_pred}')

    validation_mse = mean_squared_error(Yvalidation, Yvalidation_pred)
    #print(f'validation_mse = {validation_mse}')

    Yholdout_pred = predictor.predict(transformation.transform(Xholdout))
    #print(f'Yholdout_pred = {Yholdout_pred}')
    holdout_mse = mean_squared_error(Yholdout, Yholdout_pred)

    return (validation_mse, holdout_mse)


def average_and_std(values):
    arr = np.array(values)
    assert arr.ndim == 1 
    return (np.mean(arr), np.std(arr))



def pairs_average_and_std(pairs):
    arr = np.array([pair for pair in pairs if pair is not None])
    assert arr.ndim == 2 
    assert arr.shape[1] == 2
    pair_averages = np.mean(arr, axis=0)
    pair_stds = np.std(arr, axis=0)
    return np.hstack((pair_averages, pair_stds))


def mean_confidence_interval(data, alpha=0.95):
    (n,) = data.shape
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)/(n**0.5) # Standard error of the mean
    (a,b) = scipy.stats.t.interval(alpha, df=n-1, loc=mu, scale=sigma)
    return (a,b)
    #return f'{(a+b)/2:.3f}\u00B1{(b-a)/2:.3f}'


def print_results(list_of_validation_test_scores):
    """Print validation vs. test set means with 95% confidence intervals"""
    n = len(list_of_validation_test_scores)
    (mean_val_error, mean_test_error, val_error_std, test_error_std) = pairs_average_and_std(list_of_validation_test_scores)
    print(f'Validation error: {mean_val_error:.3f}\u00B1{1.96*val_error_std/(n**0.5):.3f}')
    print(f'Test error: {mean_test_error:.3f}\u00B1{1.96*test_error_std/(n**0.5):.3f}')


def save_figure(name, subdir=''):
    import matplotlib.pyplot as plt
    DIR = os.path.join(FIGURES_PATH, subdir).replace('.','_')
    pickler.mkdir_recursively(DIR)
    filename = os.path.join(DIR, name).replace('.','_') + '.pdf'
    print('Saving figure to', filename)
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')

