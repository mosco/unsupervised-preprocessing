# unsupervised-preprocessing

Supporting code for the paper
    "Rescaling and other forms of unsupervised preprocessing may bias cross-validation"
    by Amit Moscovich and Saharon Rosset.

By running this program, you should be able to exactly reproduce the figures in the paper.
However, using the number of repetitions used in the paper, this simulation takes
several years of single-core computation.

It is highly recommended to:
1) Do a test run with much smaller values of the constants RESCALED_LASSO_LOW_DIM_N_REPETITIONS, etc.
2) Run this program on a strong multi-core machine.
   (the code automatically parallelizes the simulations using Python's multiprocessing.Pool)

To run this, you need an installation of Python 3 with SciPy, scikit-learn and matplotlib.
The easiest way to get this setup is to download the Anaconda python distribution.

Author:
    Amit Moscovich
    amit@moscovich.org
