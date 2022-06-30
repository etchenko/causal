import numpy as np
import math
import statsmodels.api as sm
import pandas as pd
from scipy.special import expit
import scipy
import random
import matplotlib.pyplot as plt

def backdoor_adjustment(Y, A, Z, data):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    formula: list of variable names included the backdoor adjustment set
    data: pandas dataframe

    Return
    ------
    ACE: float corresponding to the causal effect
    """
    # Create the predictor variables list
    Z.insert(0,A)
    # Run the logistic regression
    log_reg = sm.GLM.from_formula(formula=f"{Y} ~ {' + '.join(map(str, Z))}", data = data).fit() #, family = sm.families.Binomial()).fit()

    # Create fragmented datasets
    df_0, df_1 = data.copy(), data.copy()
    df_0[A] = 0 
    df_1[A] = 1

    # Apply the model to the data and compute the ACE
    result_0 = log_reg.predict(df_0)
    result_1 = log_reg.predict(df_1)

    # Calculate the difference between the two fragmented datasets
    difference = result_1 - result_0

    # Compute the ACE
    ACE = difference.sum() / len(difference)
    return ACE, result_0, result_1

def compute_confidence_intervals_ace(Y, A, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []

    for i in range(num_bootstraps):
        # Resample the data with replacement
        resampled_data = data.sample(frac=1, replace = True)
        # Get the odds ratio for the resampled data
        ACE = backdoor_adjustment(Y, A, Z, resampled_data)[0]
        # Add the odds ratio to the estimates
        estimates.append(ACE)
        pass
    # Get the lower quantile
    q_low = np.quantile(estimates, Ql)
    # Get the upper quantile
    q_up = np.quantile(estimates, Qu)
    # Return the quantile estimates
    return q_low, q_up



def likelihood_test(X, Y, Z, data, printing = False):
    """
    Compute the log likelihood test on the regressions Y ~ X + Z and Y ~ Z
    X, Y are names of variables
    in the data frame. Z is a list of names of variables.

    Return float for the p-value of the log likelihood test
    """

    # Create the predictor variables list
    Z.insert(0,X)
    # Run the logistic regression on the full model
    log_reg = sm.GLM.from_formula(formula=f"{Y} ~ {' + '.join(map(str, Z))}", data = data).fit()
    # Extract the log likelihood
    ll1 = log_reg.llf
    # Run the logistic regression on the partial model
    log_reg = sm.GLM.from_formula(formula=f"{Y} ~ {' + '.join(map(str, Z[1:]))}", data = data).fit()
    # Extract the log likelihood
    ll2 = log_reg.llf

    ll_stat = -2*(ll2 - ll1)

    p_val = scipy.stats.chi2.sf(ll_stat, 2)
    # Return the odds ratio
    return p_val

def confidence_intervals_likelihood(X, Y, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals through bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """
    
    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []
    
    for i in range(num_bootstraps):
        # Implement your code here:

        # Resample the data with replacement
        resampled_data = data.sample(n = 250, replace = True)
        # Get the odds ratio for the resampled data
        likelihood = likelihood_test(X, Y, Z, resampled_data)
        # Add the odds ratio to the estimates
        estimates.append(likelihood)
        pass
    # Get the lower quantile
    q_low = np.quantile(estimates, Ql)
    # Get the upper quantile
    q_up = np.quantile(estimates, Qu)
    # Return the quantile estimates
    return q_low, q_up

def print_likelihood(X, Y, Z, data):
    '''
    Print the odds ration of Y ~ X + Z, as well as the confidence interval
    '''
    pvalue = likelihood_test(X, Y, Z, data)
    result = 'Independent' if pvalue > 0.05 else 'Dependent'
    #confidence = confidence_intervals_likelihood(X,Y,Z, data)
    print(f'Log likelihood p-value: {pvalue}, Result: {result}')

def print_ace(X,Y,Z, data):
    '''
    Print the Average Causal Effect of X on Y conditioning on Z, as well as the confidence interval
    '''
    ace = backdoor_adjustment(X,Y,Z, data)[0]
    confidence = compute_confidence_intervals_ace(X,Y,Z,data)
    print(f'ACE: {ace}, CI: {confidence}')




def generate_uniform(low, high, nums):
    '''
    Generate an array of 'nums' uniformly distributed number from low to high
    '''
    array =[0]*nums
    for i in range(nums):
        array[i] = random.uniform(low, high)
    return array

def generate_treatment_noise(p, nums):
    '''
    Generate a random array of 1 or 0, with p equaling the probabilty of a 1
    '''
    return [0 if random.random() > p else 1 for i in range(nums)]

def generate_normal(mean, sigma, nums):
    '''
    Generate an array from a normal distribution
    '''
    array = [0]*nums
    for i in range(nums):
        array[i] = np.random.normal(mean, sigma, 1)[0]
    return array

def generate_laplace(mean, decay, nums):
    '''
    Generate an array from a laplace distribution
    '''
    array = [0]*nums
    for i in range(nums):
        array[i] = np.random.laplace(mean, decay, 1)[0]
    return array


def generate_gauss(mu, sigma, nums):
    '''
    Generate an array from a gaussian distribution
    '''
    array = [0]*nums
    for i in range(nums):
        array[i] = random.gauss(mu, sigma)
    return array

def generate_treatment(data, p, noise):
    new = [0]*len(data)
    m = max(data)
    for i, item in enumerate(data):
        new[i] = 0 if item/m < p else 1
    for i, item in enumerate(new):
        if random.random() < noise:
            new[i] = 1 - item
    return new

def counter_plot(X, Y, Z, data):
    ace, r0, r1 = backdoor_adjustment(X,Y,Z, data)
    minimum = min(min(r0), min(r1))
    maximum = max(max(r0),max(r1))
    bins = np.linspace(minimum, maximum, 50)
    plt.hist(r0, bins, alpha=0.5, label="Y(a)")
    plt.hist(r1, bins, alpha=0.5, label="Y(a')")
    plt.legend(loc='upper right')
    plt.show(block = True)
