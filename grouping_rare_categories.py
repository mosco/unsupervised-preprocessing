"""
Supporting code for the paper
    "On the cross-validation bias due to unsupervised pre-processing" by Amit Moscovich and Saharon Rosset.
https://arxiv.org/abs/1901.08974v4

This module contains the simulation and graph plotting routines used in Section 5.1. "Grouping of rare categories"
that involve categorical variable pooling.

Author:
    Amit Moscovich
    amit@moscovich.org
"""
from collections import Counter, namedtuple
from numpy import array, ones, arange, repeat, mean
from numpy.random import normal, multinomial
from sklearn.utils import shuffle

import pickler
import simulations_framework
from utils import Timer, groupby

# This namedtuple holds the parameters for a single simulation of the filtered categorical regression.
# n_train - number of training samples.
# n_validation - number of validation samples. e.g. in 2-fold cross-validation n_validation is equal to n_train.
# n_holdout - number of test, or hold-out samples, used to estimate the generalization error.
#             Unlike validation samples, these samples are not involved in forming the unsupervised transformation.
# C - number of categories
# sigma - noise in the responses (y)
# M - minimum number of categories in non-rare category
ParamsRareCategoryGrouping = namedtuple('ParamsRareCategoryGrouping', 'n_train n_validation n_holdout C sigma M')

M = 4
C = 20

class DatagenCategorical:
    """
    This class generates data according to the sampling distribution of Example 1
    in the paper (grouping of rare categories)

    The mean response of every category is generated as \mu_i ~ N(0,1)

    The data samples (x,y) are generated in the following manner:
    x is a uniform draw of a category and y is a noisy measurement of the category mean.

    x ~ Unif(1,...,C)
    y = \mu_x + N(0,1)
    """
    def __init__(self, n_categories, sigma, category_weights):
        self.n_categories = n_categories
        self.mu = normal(size=n_categories)
        self.sigma = sigma

    def generate(self, n):
        n_elements_from_each_category = multinomial(n, ones(self.n_categories)/self.n_categories)
        X = repeat(arange(self.n_categories), n_elements_from_each_category)
        Y = self.mu[X] + normal(loc=0, scale=self.sigma, size=n)
        return shuffle(X, Y)


class CategoricalVariableTransformer:
    """
    This preliminary unsupervised transformation lumps together all samples from categories
    that appear < M times into a "rare" category.
    """
    def __init__(self, min_category_elements, rare_category):
        self.min_category_elements = min_category_elements
        self.rare_category = rare_category

    def fit(self, X):
        assert X.ndim == 1
        assert self.rare_category not in X
        self.categories = set()
        for (category, count) in Counter(X).items():
            if count >= self.min_category_elements:
                self.categories.add(category)

        return self

    def transform(self, X):
        return array([x if x in self.categories else self.rare_category for x in X])


class CategoryMeanRegressor:
    """
    Predict the response associated with a category by taking the mean response 
    of all samples from that category in the training set.
    """
    def __init__(self, default_prediction, overrides_dict = {}):
        self.default_prediction = default_prediction
        self.overrides_dict = overrides_dict

    def fit(self, X, Y):
        self.category_means = {}
        ys_grouped_by_x = groupby(zip(X,Y), keyfunc=lambda x_y: x_y[0], mapfunc=lambda x_y: x_y[1])
        for (x, ys) in ys_grouped_by_x.items():
            self.category_means[x] = mean(ys)
        for (k,v) in self.overrides_dict.items():
            self.category_means[k] = v

    def predict(self, X):
        return array([self.category_means[x] if x in self.category_means else self.default_prediction for x in X])


def simulate(params):
    "A single simulation run of Example 1 from the paper"
    return simulations_framework.simulate_validation_vs_holdout_mse(params.n_train, params.n_validation, params.n_holdout,
            data_generator = DatagenCategorical(params.C, params.sigma, ones(params.C)/params.C),
            transformation = CategoricalVariableTransformer(params.M, rare_category=-1),
            predictor = CategoryMeanRegressor(default_prediction=0.0, overrides_dict={-1: 0.0}))


def run_precalc(filename, sizes, C, sigma, M, n_repetitions):
    "Run multiple repetitions of simulate() and save the results in a pickle file."
    job_params = [ParamsRareCategoryGrouping(n_train, n_validation, n_holdout, C, sigma, M) for (n_train,n_validation,n_holdout) in sizes]
    simulations_framework.parallel_montecarlo(filename, simulate, simulations_framework.pairs_average_and_std, job_params, n_repetitions, seed=7)


def precalc_all(REPS):
    """
    Precalculate all results needed to produce the figures for Example 1 in the paper.

    Each of these simulations takes around 26 hours per 1 million iterations on a single core of a 2016 Intel Xeon
    """
    for sigma in [0.25, 1.5]:
        print('-'*60)

        N_RANGE = arange(5,105,5)

        filename = f'categorical_K2_C{C}_sigma{sigma:.2f}_M{M}'.replace('.','_')
        with Timer(f'{filename} ({REPS} repetitions)'):
            run_precalc(filename, [(n,n,n) for n in N_RANGE], C, sigma, M, REPS)

        filename = f'categorical_LOO_C{C}_sigma{sigma:.2f}_M{M}'.replace('.','_')
        with Timer(f'{filename} ({REPS} repetitions)'):
            run_precalc(filename, [(n,1,n) for n in N_RANGE], C, sigma, M, REPS)


def plot_test_vs_validation_set(output_name, C, sigma, M, xlim=None, ylim=None, xticks=None, yticks=None):
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

    d = pickler.load(f'categorical_K2_C{C}_sigma{sigma:.2f}_M{M}'.replace('.','_'))
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0-', label=r'$\mathrm{e}_{\mathrm{val}}(m=n)$')
    plt.plot(x_values, d.results[:,1], 'C1-', label=r'$\mathrm{e}_{\mathrm{gen}}(m=n)$')

    d = pickler.load(f'categorical_LOO_C{C}_sigma{sigma:.2f}_M{M}'.replace('.','_'))
    x_values = [x.n_train for x in d.xs]
    plt.plot(x_values, d.results[:,0], 'C0--', label=r'$\mathrm{e}_{\mathrm{val}}(m=1)$')
    plt.plot(x_values, d.results[:,1], 'C1--', label=r'$\mathrm{e}_{\mathrm{gen}}(m=1)$')

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
    simulations_framework.save_figure(output_name + f'_reps{d.n_repetitions}')


def plot_all():
    for sigma in [0.25, 1.5]:
        print('-'*60)
        plot_test_vs_validation_set(f'categorical_C{C}_sigma{sigma:.2f}_M{M}', C, sigma, M, xlim=[5,100], xticks=arange(10,110,10))
