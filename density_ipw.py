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
from scipy import stats
import pylab
pd.options.mode.chained_assignment = None  # default='warn'


#
# Calculate the counterfactual distributions using a regression to find the propensity weighting
#
def practical_counterfactual_distributions(A, C, Y, data, norm = True, trim = True):
    log_reg = sm.GLM.from_formula(formula=f"{A} ~ {'+'.join(C)}", data = data, family = sm.families.Binomial()).fit()
    log_propensity = log_reg.predict(data)
    data['log_p'] = np.where(data[A] == 0, 1 - log_propensity, log_propensity)
    return counterfactual_distributions(A, Y, 'log_p', data, norm, trim)

#
# Calculate the counterfactual distributions using theoretical weighting
#
def theoretical_counterfactual_distributions(A, T, Y, data, norm = True, trim = True):
    return counterfactual_distributions(A, Y, T, data, norm, trim)

#
# Calculate counterfactual distribbution
#
def counterfactual_distributions(A, Y, prop, data, norm, trimming):
    '''
    Calculate counterfactual distribbution
    '''
    if norm:
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
    data = data.copy()
    for i in range(data[prop].last_valid_index(), data[prop].first_valid_index(), -1):
        if data[prop].iloc[i] <= a or data[prop].iloc[i] >= 1 - a:
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

def kernel_estimation(A, C, Y, data):
    log_reg = sm.GLM.from_formula(formula=f"{A} ~ {'+'.join(C)}", data = data, family = sm.families.Binomial()).fit()
    log_propensity = log_reg.predict(data)
    data['log_p'] = np.where(data[A] == 0, 1 - log_propensity, log_propensity)

    minimum = data[Y].min()
    maximum = data[Y].max()
    #data[Y] = (data[Y] - minimum)/maximum

    data = trim(data, 'log_p')

    a1 = data[data[A] == 1]
    a0 = data[data[A] == 0]

    prop1 = 1/a1['log_p']
    prop0 = 1/a0['log_p']

    propensity_weighting_1 = prop1/(prop1.sum())
    propensity_weighting_0 = prop0/(prop0.sum())


    a1_resample = np.random.choice(a1[Y], int(len(a1)/2), replace = True, p = propensity_weighting_1)
    a0_resample = np.random.choice(a0[Y], int(len(a0)/2), replace = True, p = propensity_weighting_0)

    #a0_resample, a1_resample = theoretical_counterfactual_distributions('A', 'theor', 'Y', data, norm = True, trim = False)


    x = np.linspace(minimum,maximum,100)
    # First Graph
    kernel2 = stats.gaussian_kde(a1[Y], 'scott')
    kernel3 = stats.gaussian_kde(a0[Y], 'scott')

    pylab.plot(x,kernel2(x),"g", label = "Treatment") # distribution function
    pylab.plot(x,kernel3(x),"m", label = 'No Treatment') # distribution function
    pylab.hist(a1[Y],density=True,alpha=.3, color = "g") # histogram
    pylab.hist(a0[Y],density=True,alpha=.3, color = "m") # histogram
    pylab.legend(fontsize=15, title_fontsize=20)
    pylab.ylabel('p(Y | A)', fontsize = 15)
    pylab.xlabel('Outcome', fontsize = 15)
    #pylab.title("Before")
    pylab.show()

    kernel2 = stats.gaussian_kde(a1_resample, 'scott')
    kernel3 = stats.gaussian_kde(a0_resample, 'scott')


    #pylab.plot(x,kernel1(x),"r", label = "Y(a')") # distribution function
    #pylab.plot(x,kernel0(x),"b", label = 'Y(a)') # distribution function
    pylab.plot(x,kernel2(x),"g", label = "Treatment") # distribution function
    pylab.plot(x,kernel3(x),"m", label = 'No Treatment') # distribution function
    pylab.hist(a1_resample,density=1,alpha=.3, color = "g") # histogram
    pylab.hist(a0_resample,density=1,alpha=.3, color = "m") # histogram
    pylab.legend(fontsize=15, title_fontsize=20)
    pylab.ylabel('p(Y | do(A))', fontsize = 15)
    pylab.xlabel('Outcome', fontsize = 15)
    #pylab.title("After IPW")
    pylab.show()
    


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
    nums = 10000
    data = pd.DataFrame()
    data['C'] =  sf.generate_uniform(0, 1, nums)
    mi, ma = min(data['C']), max(data['C'])
    ran = ma - mi
    arr = [0]*nums
    for i in range(nums):
        p = (data['C'][i] - mi)/ran
        r = random.random()
        arr[i] = 0 if r < p else 1
    data['A'] = arr
    #data['theor'] = np.where(data['A'] == 0, np.where(data['C'] > 0, 0.25, 0.75), np.where(data['C'] > 0, 0.75, 0.25))
    data['Y'] = sf.generate_normal(0, 1, nums)
    for i in range(nums):
        data['Y'][i] += random.gauss(0, abs((data['C'][i]*5))) if data['A'][i] == 0 else random.gauss(0, abs((data['C'][i])*10))
    #for i in range(len(data)):
    #    data['Y'][i] = data['Y'][i] + (1 if data['A'][i] < data['C'][i] + 1 else 0)

    #check_distribution_change(data, nums)
    #check_y_change(data, nums)
    #check_no_change(data, nums)
    kernel_estimation('A',['C'],'Y', data)


#
# Testing of code and of conjecture
#
def main():
    test_generated()
    #nsw_randomized = pd.read_csv("Data/nsw_randomized.txt")
    #nsw_observational = pd.read_csv("Data/nsw_observational.txt")

    #covariates = ["age","educ","black","hisp","marr","nodegree","re74","re75"]
    #interaction = ["age","educ","black","hisp","marr","nodegree","re74","re75","black*nodegree","black*treat","black*educ","hisp*nodegree","hisp*treat","hisp*educ","treat*nodegree","treat*educ"]
    #output = practical_counterfactual_distributions('treat', covariates, 're78', nsw_observational)
    #output2 = practical_counterfactual_distributions('treat', covariates, 're78', nsw_observational)
    #print(output[0])
    #print(output[1])
    #plot_distributions(output, 'Randomized')
    
    #kernel_estimation('treat',covariates,'re78',nsw_observational)
    #test_generated()
    '''
    for i in range(10):
        nsw_observational = pd.read_csv("Data/nsw_observational.txt")
        output = practical_counterfactual_distributions('treat', covariates, 're78', nsw_observational)
        plot_distributions(output, 'Outliers present')
        print(f"W outliers: {mmd.ci_mmd(output[0], output[1])}")
        output2 = (remove_outliers(output[0]), remove_outliers(output[1]))
        plot_distributions(output2, "Outliers Removed")
        print(f"No outliers: {mmd.ci_mmd(output2[0], output2[1])}")
    '''
    '''
    minimum = nsw_randomized['re78'].min()
    maximum = nsw_randomized['re78'].max() - minimum
    nsw_randomized['re78'] = (nsw_randomized['re78'] - minimum)/maximum

    nsw_1 = nsw_randomized[nsw_randomized['treat'] == 1]
    nsw_0 = nsw_randomized[nsw_randomized['treat'] == 0]
    kernel1 = stats.gaussian_kde(nsw_1['re78'], 'scott')
    kernel0 = stats.gaussian_kde(nsw_0['re78'], 'scott')

    x = np.linspace(0,1,100)
    pylab.plot(x,kernel1(x),"r", label = "Y(a')") # distribution function
    pylab.plot(x,kernel0(x),"b", label = 'Y(a)') # distribution function
    pylab.hist(nsw_1['re78'],density=1,alpha=.3, color = "m", label = "Y(a')") # histogram
    pylab.hist(nsw_0['re78'],density=1,alpha=.3, color = "g", label = "Y(a)") # histogram
    pylab.legend(loc='upper right')
    pylab.title("Counterfactual Densities")
    pylab.show()

    #plot_distributions(output2, 'Observational')
    #print(mmd.ci_mmd(output2[0], output2[1]))
    '''


if __name__ == "__main__":
    main()