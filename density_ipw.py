from calendar import prcal
from math import isqrt
import sim_functions as sf
import statsmodels.api as sm
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import mmd
import warnings
pd.options.mode.chained_assignment = None  # default='warn'

#
# Calculate the counterfactual distributions using a regression to find the propensity weighting
#
def practical_counterfactual_distributions(A, C, Y, data, trim = True):
    log_reg = sm.GLM.from_formula(formula=f"{A} ~ {'+'.join(C)}", data = data, family = sm.families.Binomial()).fit()
    log_propensity = log_reg.predict(data)
    data['log_p'] = np.where(data[A] == 0, 1 - log_propensity, log_propensity)
    return counterfactual_distributions(A, Y, 'log_p', data, trim)

#
# Calculate the counterfactual distributions using theoretical weighting
#
def theoretical_counterfactual_distributions(A, T, Y, data, trim = True):
    return counterfactual_distributions(A, Y, T, data, trim)

#
# Calculate counterfactual distribbution
#
def counterfactual_distributions(A, Y, prop, data, trimming):
    minimum = data[Y].min()
    maximum = data[Y].max() - minimum
    data[Y] = (data[Y] - minimum)/maximum

    if trimming:
        data = trim(data, prop)

    a1 = data[data[A] == 1]
    a0 = data[data[A] == 0]

    prop1 = 1/a1[prop]
    prop0 = 1/a0[prop]

    propensity_weighting_1 = prop1/(prop1.sum())
    propensity_weighting_0 = prop0/(prop0.sum())


    a1_resample = np.random.choice(a1[Y], int(len(a1)/2), replace = True, p = propensity_weighting_1)
    a0_resample = np.random.choice(a0[Y], int(len(a0)/2), replace = True, p = propensity_weighting_0) 
    return a0_resample, a1_resample

def trim(data, prop, a = 0.1):
    for i in range(data[prop].last_valid_index(), data[prop].first_valid_index() - 1, -1):
        if data[prop][i] <= a or data[prop][i] >= 1 - a:
            # Trim outliers
            data.drop(i, inplace = True)
    return data

def remove_outliers(data):
    q3 = np.quantile(data, 0.75)
    q1 = np.quantile(data, 0.25)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    data = list(filter(lambda x: x < upper and x > lower, data))
    return data


# 
# Plot the distributions
#
def plot_distributions(distributions, title):
    plt.hist(distributions[0], alpha=0.5, label="Y(a)")
    plt.hist(distributions[1], alpha=0.5, label="Y(a')")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show(block = True)

def check_distribution_change(data, nums):
    data['Y2'] = np.where(data['A'] == 0, sf.generate_laplace(0, 3, nums), sf.generate_normal(0, 3, nums)) + data['C']

    theor_dist = theoretical_counterfactual_distributions('A', 'theor', 'Y2', data)
    plot_distributions(theor_dist, "(Theor) A changes distribution of Y")
    prac_dist = practical_counterfactual_distributions('A', ['C'], 'Y2', data)
    plot_distributions(prac_dist, "(Prac) A changes distribution of Y")

    print("Theoretical and practical MMD when only distribution is changed")
    print(mmd.ci_mmd(theor_dist[0], theor_dist[1]))
    print(mmd.ci_mmd(prac_dist[0], prac_dist[1]))

def check_y_change(data, nums):
    data['Y'] = data['C'] + 2*data['A'] + sf.generate_uniform(-1, 1, nums)

    theor_dist = theoretical_counterfactual_distributions('A', 'theor', 'Y', data)
    plot_distributions(theor_dist, "(Theor) A changes value of Y)")
    prac_dist = practical_counterfactual_distributions('A', ['C'], 'Y', data)
    plot_distributions(prac_dist, "(Prac) A changes value of Y)")

    print("Theoretical and practical MMD when A influences Y")
    print(mmd.ci_mmd(theor_dist[0], theor_dist[1]))
    print(mmd.ci_mmd(prac_dist[0], prac_dist[1]))

def check_no_change(data, nums):
    data['Y1'] = data['C']  + sf.generate_uniform(-1, 1, nums) 

    theor_dist = theoretical_counterfactual_distributions('A', 'theor', 'Y1', data)
    plot_distributions(theor_dist, "(Theor) A does not change value of Y)")
    prac_dist = practical_counterfactual_distributions('A', ['C'], 'Y1', data)
    plot_distributions(prac_dist, "(Prac) A does not change value of Y)")

    print("Theoretical and practical MMD when A does not influence Y")
    print(mmd.ci_mmd(theor_dist[0], theor_dist[1]))
    print(mmd.ci_mmd(prac_dist[0], prac_dist[1]))

def test_generated():
    nums = 2000
    data = pd.DataFrame()
    data['C'] = sf.generate_normal(0, 1, nums)
    arr = [0]*nums
    for i in range(nums):
        p = random.random()
        if data['C'][i] > 0:
            arr[i] = 0 if p < 0.7 else 1
        else:
            arr[i] = 0 if p < 0.3 else 1
    data['A'] = arr
    data['theor'] = np.where(data['A'] == 0, np.where(data['C'] > 0, 0.7, 0.3), np.where(data['C'] > 0, 0.3, 0.7))

    check_distribution_change(data, nums)
    check_y_change(data, nums)
    check_no_change(data, nums)


#
# Testing of code and of conjecture
#
def main():
    nsw_randomized = pd.read_csv("Data/nsw_randomized.txt")
    nsw_observational = pd.read_csv("Data/nsw_observational.txt")

    covariates = ["age","educ","black","hisp","marr","nodegree","re74","re75"]
    #interaction = ["age","educ","black","hisp","marr","nodegree","re74","re75","black*nodegree","black*treat","black*educ","hisp*nodegree","hisp*treat","hisp*educ","treat*nodegree","treat*educ"]
    #output = practical_counterfactual_distributions('treat', covariates, 're78', nsw_observational)
    #output2 = practical_counterfactual_distributions('treat', covariates, 're78', nsw_observational)
    #print(output[0])
    #print(output[1])
    #plot_distributions(output, 'Randomized')
    for i in range(10):
        nsw_observational = pd.read_csv("Data/nsw_observational.txt")
        output = practical_counterfactual_distributions('treat', covariates, 're78', nsw_observational)
        plot_distributions(output, 'Outliers present')
        print(f"W outliers: {mmd.ci_mmd(output[0], output[1])}")
        output2 = (remove_outliers(output[0]), remove_outliers(output[1]))
        plot_distributions(output2, "Outliers Removed")
        print(f"No outliers: {mmd.ci_mmd(output2[0], output2[1])}")

    #plot_distributions(output2, 'Observational')
    #print(mmd.ci_mmd(output2[0], output2[1]))


if __name__ == "__main__":
    main()