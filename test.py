from typing import Callable
import pandas as pd
import numpy as np
import sim_functions as sf
from scipy.stats import wasserstein_distance

def tester(func):
    # Run tests on the given function, which should take as input two numpy arrays
    print("Nothing Yet")

def real_data(func):
    # Import data
    nsw_observational = pd.read_csv("Data/nsw_observational.txt")
    covariates = ["age","educ","black","hisp","marr","nodegree","re74","re75"]
    #TODO: Run test to find statistic and p-value, maybe a range

def diff_disst(func):
    #TODO: Go through a few different distributions and compare them to each other
    print("Nothing Yet")

def ACE(func):
    #TODO: Test data on just differing amounts of causal effect from the same / maybe differing distributions
    print("Nothing Yet")

def wass_conf(x: np.array, y: np.array) -> tuple[float, float]:
    return func_conf(x, y, wasserstein_distance)


def func_conf(x: np.array, y: np.array, distance_function: Callable, num_bootstraps = 1000) -> tuple[float, float]:
    """ Find the statistic and p-value of 2 arrays using a given distance function

    Keyword Arguments:
    x -- A numpy array
    y -- A second numpy array
    distance_function -- A distance function which takes as input to numpy arrays
    num_bootstraps -- The number of bootstrap runs to find the p-value (default 1000)
    """
    # Calculate distance
    dist = distance_function(x, y)
    combined = np.append(x, y)
    count = 0
    # Boostrap from combined dataset
    for i in range(num_bootstraps):
        # Resample
        new_data_x = np.random.choice(combined, len(x))
        new_data_y = np.random.choice(combined, len(y))
        # Find statistics
        dist2 = distance_function(new_data_x, new_data_y)
        # Check if higher than found distance
        if dist2 >= dist:
            count += 1
    # Comopute p-value
    pvalue = count / num_bootstraps if count > 0 else 1 / num_bootstraps
    return dist, pvalue


def main():
    # Test whether it differentiates gaussian and laplacian distributions
    gauss = sf.generate_gauss(0, 1, 1000)
    lap = sf.generate_laplace(0, 1/2, 1000)


    print(wass_conf(gauss, lap))


if __name__ == '__main__':
    main()