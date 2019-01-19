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


GROUPING_RARE_CATEGORIES_N_REPETITIONS = 10**3

RESCALED_LASSO_LOW_DIM_N_REPETITIONS = 10**3
RESCALED_LASSO_HIGH_DIM_N_REPETITIONS = 10**2


print('\nWarning! running these computations on a single core will take more than a year.')
print('It is recommended to first try to run the computations with a much lower number of repetitions than the defaults here.\n')
print('==== Running precalculations for the figures of Example 1 ====')
grouping_rare_categories.precalc_all(GROUPING_RARE_CATEGORIES_N_REPETITIONS)

print('==== Running precalculations for the figures of Example 2 ====')
rescaled_lasso.precalc_all(RESCALED_LASSO_LOW_DIM_N_REPETITIONS, RESCALED_LASSO_HIGH_DIM_N_REPETITIONS)

print('================================================================')
print('==== FINISED CALCULATIONS ======================================')
print('================================================================')

print('==== Generating all figures from precalculations ====')
print("(if this step fails, you don't need to rerun the lengthy precalculations)")
grouping_rare_categories.plot_all()
rescaled_lasso.plot_all()

