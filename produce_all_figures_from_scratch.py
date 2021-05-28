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
import variable_selected_linear_regression_realdata
import model_selection_cv_pipeline
from utils import Timer


GROUPING_RARE_CATEGORIES_N_REPETITIONS = 10**7

VARIABLE_SELECTED_LINREG_REPS_LOWDIM = 10**6
VARIABLE_SELECTED_LINREG_REPS_HIGHDIM = 10**6

VARIABLE_SELECTED_LINREG_REPS_SUPERCONDUCTIVITY = 10**6

RESCALED_LASSO_LOW_DIM_N_REPETITIONS = 10**7
RESCALED_LASSO_HIGH_DIM_N_REPETITIONS = 10**6

CV_PIPELINE_LOW_DIM_N_REPETITIONS = 10**5
CV_PIPELINE_HIGH_DIM_N_REPETITIONS = 10**5


def precalc_all():
    separator = '\n' + '-'*100 + '\n' + 'X'*100 + '\n' + '-'*100 + '\n'

    print(separator)

    with Timer('==== Running precalculations for Figure 4 in Section 5.1 ==========\n==== (pooling followed by category mean regression) ========='):
        grouping_rare_categories.precalc_all(GROUPING_RARE_CATEGORIES_N_REPETITIONS)

    print(separator)

    with Timer('==== Running precalculations for Figure 1 and Figure 3 in Section 4.1 =========\n==== (variance-based feature selection followed by linear regression) ========='):
        variable_selected_linear_regression.precalc_all(VARIABLE_SELECTED_LINREG_REPS_LOWDIM, VARIABLE_SELECTED_LINREG_REPS_HIGHDIM)

    print(separator)

    with Timer('==== Running precalculations for Figure 2 in Section 4.2 =========\n====  (variance-based feature selection followed by linear regression on the superconductivity dataset) ========='):
        variable_selected_linear_regression_realdata.precalc_all(VARIABLE_SELECTED_LINREG_REPS_SUPERCONDUCTIVITY)

    print(separator)

    with Timer('==== Running precalculations for Figure 5 in Section 5.2 =========\n==== (feature rescaling followed by Lasso linear regression) ========='):
        rescaled_lasso.precalc_all(RESCALED_LASSO_LOW_DIM_N_REPETITIONS, RESCALED_LASSO_HIGH_DIM_N_REPETITIONS)

    print(separator)

    with Timer('==== Running precalculations for Figure 6 in Section 6 =========\n==== (generalization error following correct vs. incorrect model selection) ========='):
        model_selection_cv_pipeline.precalc_all(CV_PIPELINE_LOW_DIM_N_REPETITIONS, CV_PIPELINE_HIGH_DIM_N_REPETITIONS)

    print(separator)


def plot_all():
    grouping_rare_categories.plot_all()
    variable_selected_linear_regression.plot_all()
    variable_selected_linear_regression_realdata.plot_all()
    rescaled_lasso.plot_all()
    model_selection_cv_pipeline.plot_all()


def main():
    print('\nWarning! running these computations on a single core will take more than a year.')
    print('It is recommended to first try to run the computations with a much lower number of repetitions than the defaults above (say 1/1000).')
    precalc_all()

    print('=================================================================')
    print('==== FINISHED CALCULATIONS ======================================')
    print('=================================================================')

    print('==== Generating all figures from precalculations ====')
    print("(if this step fails, you don't need to rerun the lengthy precalculations)")
    plot_all()


if __name__ == '__main__':
    main()
