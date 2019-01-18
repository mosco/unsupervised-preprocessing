By running `produce_all_figures_from_scratch.py`, you should be able to exactly reproduce the figures in the paper.

Using the default number of repetitions (as used in the paper), this simulation takes 1-2 years on a single core. Therefore it is highly recommended to:
1. Do a test run with much smaller values of the constants RESCALED_LASSO_LOW_DIM_N_REPETITIONS, etc.
1. Run this program on a strong multi-core machine. The code automatically parallelizes the simulations using Python's multiprocessing.Pool.

# Prerequisites

An installation of Python 3 with SciPy, scikit-learn, mkl and mkl_random modules.
The easiest way to get this setup is to download the Anaconda python distribution.

# Contact

Feel free to shoot me an email.

Amit Moscovich
amit@moscovich.org
