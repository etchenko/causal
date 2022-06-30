from calendar import prcal
import sim_functions as sf
import statsmodels.api as sm
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import mmd
pd.options.mode.chained_assignment = None  # default='warn'

#
# Calculate the counterfactual distributions using a regression to find the propensity weighting
#
def practical_counterfactual_distributions(A, C, Y, data):
    log_reg = sm.GLM.from_formula(formula=f"{A} ~ {'+'.join(C)}", data = data, family = sm.families.Binomial()).fit()
    log_propensity = log_reg.predict(data)
    data['log_p'] = np.where(data['A'] == 0, 1 - log_propensity, log_propensity)
    return counterfactual_distributions(A, Y, 'log_p', data)

#
# Calculate the counterfactual distributions using theoretical weighting
#
def theoretical_counterfactual_distributions(A, T, Y, data):
    return counterfactual_distributions(A, Y, T, data)

#
# Calculate counterfactual distribbution
#
def counterfactual_distributions(A, Y, prop, data):
    a1 = data[data[A] == 1]
    a0 = data[data[A] == 0]

    prop1 = 1/a1[prop]
    prop0 = 1/a0[prop]
    propensity_weighting_1 = prop1/(prop1.sum())
    propensity_weighting_0 = prop0/(prop0.sum())

    a1_resample = np.random.choice(a1[Y], int(len(a1)/2), replace = True, p = propensity_weighting_1)
    a0_resample = np.random.choice(a0[Y], int(len(a0)/2), replace = True, p = propensity_weighting_0)  
    return a0_resample, a1_resample

# 
# Plot the distributions
#
def plot_distributions(distributions, title):
    plt.hist(distributions[0], alpha=0.5, label="Y(a)")
    plt.hist(distributions[1], alpha=0.5, label="Y(a')")
    plt.legend(loc='upper right')
    plt.title(title)
    #plt.show(block = True)

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


#
# Testing of code and of conjecture
#
def main():
    # Generate the data
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


if __name__ == "__main__":
    main()