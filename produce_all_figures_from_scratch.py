#!/usr/bin/env python3
#
# Supporting code for the paper
#     "Rescaling and other forms of unsupervised preprocessing may bias cross-validation"
#     by Amit Moscovich and Saharon Rosset.
# 
# By running this program, you should be able to exactly reproduce the figures in the paper.
# However, using the number of repetitions used in the paper, this simulation takes
# several years of single-core computation.
#
# It is highly recommended to:
# 1) Do a test run with much smaller values of the constants RESCALED_LASSO_LOW_DIM_N_REPETITIONS, etc.
# 2) Run this program on a strong multi-core machine.
#    (the code automatically parallelizes the simulations using Python's multiprocessing.Pool)
#
# Author:
#     Amit Moscovich
#     amit@moscovich.org

import grouping_rare_categories
import rescaled_lasso
import variable_selected_linear_regression
from utils import Timer


GROUPING_RARE_CATEGORIES_N_REPETITIONS = 10**7

VARIABLE_SELECTED_LIN_REG_REPS_LOWDIM = 10**6
VARIABLE_SELECTED_LIN_REG_REPS_HIGHDIM = 10**6

RESCALED_LASSO_LOW_DIM_N_REPETITIONS = 10**6
RESCALED_LASSO_HIGH_DIM_N_REPETITIONS = 10**6


def precalc_all():
    with Timer('==== Running precalculations for the figures of Example 1 ===='):
        grouping_rare_categories.precalc_all(GROUPING_RARE_CATEGORIES_N_REPETITIONS)

    with Timer('==== Running precalculations for the figures of Example 2 ===='):
        variable_selected_linear_regression.precalc_all(VARIABLE_SELECTED_LIN_REG_REPS_LOWDIM, VARIABLE_SELECTED_LIN_REG_REPS_HIGHDIM)

    with Timer('==== Running precalculations for the figures of Example 3 ===='):
        rescaled_lasso.precalc_all(RESCALED_LASSO_LOW_DIM_N_REPETITIONS, RESCALED_LASSO_HIGH_DIM_N_REPETITIONS)


def plot_all():
    grouping_rare_categories.plot_all()
    rescaled_lasso.plot_all()
    variable_selected_linear_regression.plot_all()


def main():
    print('\nWarning! running these computations on a single core will take more than a year.')
    print('It is recommended to first try to run the computations with a much lower number of repetitions than the defaults here.\n')
    precalc_all()

    print('================================================================')
    print('==== FINISED CALCULATIONS ======================================')
    print('================================================================')

    print('==== Generating all figures from precalculations ====')
    print("(if this step fails, you don't need to rerun the lengthy precalculations)")
    plot_all()


if __name__ == '__main__':
    main()
