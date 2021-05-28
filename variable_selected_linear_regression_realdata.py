"""
Supporting code for the paper
    "On the cross-validation bias due to unsupervised pre-processing" by Amit Moscovich and Saharon Rosset.
https://arxiv.org/abs/1901.08974v4

This module contains the simulation and plotting routines used to generate the results on the superconductivity
dataset, Figure 2, Section "4.2. Experiments on a real dataset".

Author:
    Amit Moscovich
    amit@moscovich.org
"""
from collections import namedtuple, Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from simulations_framework import parallel_montecarlo, simulate_validation_vs_holdout_mse, average_and_std, pairs_average_and_std, save_figure, print_results
import variable_selected_linear_regression
import pickler
from utils import Timer

RANDOM_SEED = 42


ParamsGenericLinearRegression = namedtuple('ParamsGenericLinearRegression', 'datagen n_train n_validation n_holdout M')



def read_dataset_superconductivity():
    df = pd.read_csv('superconductivity/train.csv')
    arr = df.to_numpy()
    assert arr.shape == (21263, 82)

    X = arr[:,:81] # Column 25 is identically zero!
    Y = arr[:,81]
    return (X, Y)


class DatagenSubsample:
    def __init__(self, X, Y):
        (n,p) = X.shape
        assert Y.shape == (n,)
        self.n = n
        self.X = X
        self.Y = Y

    def generate(self, k):
        indices = np.random.choice(self.n, k, replace=False)
        return (self.X[indices,:], self.Y[indices])


def superconductivity_selected_variables_hist(n_sampled_rows, n_selected_vars, reps):
    (X, Y) = read_dataset_superconductivity()
    (n, p) = X.shape
    assert n_sampled_rows <= n
    assert n_selected_vars <= p

    variable_selector = variable_selected_linear_regression.TopKVarianceVariableSelector(n_selected_vars)

    selected_variable_tuples = []
    variable_masks = []
    for i in range(reps):
        X_row_subset = X[np.random.choice(n, n_sampled_rows, replace=False),:]
        variable_selector.fit(X_row_subset)
        selected_variable_tuples.append(tuple(sorted(variable_selector._selected_variables)))
        variable_masks.append(variable_selector._mask)


    return (Counter(selected_variable_tuples), np.mean(variable_masks, axis=0))


def simulate(params):
    try:
        res = simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
                data_generator=params.datagen,
                transformation=variable_selected_linear_regression.TopKVarianceVariableSelector(params.M),
                predictor=LinearRegression())
    except np.linalg.LinAlgError as e:
        print('Caught numpy.linalg.LinAlgError! Ignoring result')
        print(e)
        return None
    return res


def precalc(X, Y, datasetname, n_range, M, normalize, reps):
    if normalize:
        Xnormalized = (X / np.std(X,axis=0))
        np.random.seed(RANDOM_SEED)
        noise = np.random.normal(size=X.shape, scale=1e-6) # To make the dataset better conditioned
        #datagen = DatagenSubsample(Xnormalized+noise, Y)
        datagen = DatagenSubsample(Xnormalized, Y)
    else:
        datagen = DatagenSubsample(X, Y)

    fn_K2 = f'variable_selected_linear_regression_{datasetname}_K2_M{M}_normalize{normalize}'
    params_K2 = [ParamsGenericLinearRegression(datagen=datagen, n_train=n, n_validation=n, n_holdout=n, M=M) for n in n_range]

    fn_LOO = f'variable_selected_linear_regression_{datasetname}_LOO_M{M}_normalize{normalize}'
    params_LOO = [ParamsGenericLinearRegression(datagen=datagen, n_train=n, n_validation=1, n_holdout=n, M=M) for n in n_range]

    for (filename, job_params) in [(fn_K2, params_K2), (fn_LOO, params_LOO)]:
        with Timer(f'{filename} ({reps} repetitions)'):
            parallel_montecarlo(filename, simulate, pairs_average_and_std, job_params, reps, seed=RANDOM_SEED)





def precalc_all(reps):
    (X,Y) = read_dataset_superconductivity()
    precalc(X, Y, 'superconductivity', range(20,  65,  5), 10, True, reps)
    precalc(X, Y, 'superconductivity', range(60,  130,  10), 30, True, reps)


def plot_test_vs_validation_set(subdir, filename_prefix, M, normalize, xlim=None, ylim=None, xticks=None, yticks=None):
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


    d = pickler.load(f'{filename_prefix}_K2_M{M}_normalize{normalize}')
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0-', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{val}}(m=n)$')
    plt.plot(x_values, d.results[:,1], 'C1-', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{gen}}(m=n)$')

    d = pickler.load(f'{filename_prefix}_LOO_M{M}_normalize{normalize}')
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0--', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{val}}(m=1)$')
    plt.plot(x_values, d.results[:,1], 'C1--', linewidth=1.5, label=r'$\mathrm{e}_{\mathrm{gen}}(m=1)$')

    plt.xlabel('$n$')
    plt.ylabel('MSE')
    plt.xlim(xlim if xlim is not None else [min(x_values), max(x_values)])
    if ylim is not None:
        plt.ylim(ylim)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    
    #ax.set_yscale('log')
    plt.legend(loc='best')
    output_name = f'{filename_prefix}_M{M}_normalize{normalize}_reps{d.n_repetitions}'
    save_figure(output_name, subdir=subdir)


def plot_all():
    plot_test_vs_validation_set('superconductivity', 'variable_selected_linear_regression_superconductivity', 10, True)
    plot_test_vs_validation_set('superconductivity', 'variable_selected_linear_regression_superconductivity', 30, True)


