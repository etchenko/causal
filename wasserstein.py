from ensurepip import bootstrap
from turtle import pd
from scipy.stats import wasserstein_distance
import sim_functions as sf
import pandas as pd
import matplotlib.pyplot as plt
import density_ipw as di
import random
import numpy as np
import pylab
import statsmodels.api as sm

def pvalue(stat, a1, a0, weight1, weight2, bootstrap = 1000):
    combined = np.append(a1, a0)
    count = 0
    for i in range(bootstrap):
        new_data_x = np.random.choice(combined, len(a1))
        new_data_y = np.random.choice(combined, len(a0))
        wass = wasserstein_distance(new_data_x, new_data_y)
        if wass > stat:
            stat




nums = 1000

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
ran = np.arange(-0.5, 0.5, 0.01)
vals = []
mini = []
maxi = []
for i in ran:
    data['Y'] = data['C'] + i*data['A'] + sf.generate_uniform(-1, 1, nums)
    log_reg = sm.GLM.from_formula(formula=f"A ~ C", data = data, family = sm.families.Binomial()).fit()
    log_propensity = log_reg.predict(data)
    data['log_p'] = np.where(data['A'] == 0, 1 - log_propensity, log_propensity)
    a1 = data[data['A'] == 1]
    a0 = data[data['A'] == 0]
    weights1 = 1/a1['log_p']
    weights0 = 1/a0['log_p']
    vals.append(wasserstein_distance(a1['Y'], a0['Y'], weights1, weights0))
    arr = []
    for j in range(100):
        data['Y'] = data['C'] + i*data['A'] + sf.generate_uniform(-1, 1, nums)
        log_reg = sm.GLM.from_formula(formula=f"A ~ C", data = data, family = sm.families.Binomial()).fit()
        log_propensity = log_reg.predict(data)
        data['log_p'] = np.where(data['A'] == 0, 1 - log_propensity, log_propensity)
        a1 = data[data['A'] == 1]
        a0 = data[data['A'] == 0]
        weights1 = 1/a1['log_p']
        weights0 = 1/a0['log_p']
        arr.append(wasserstein_distance(a1['Y'], a0['Y'], weights1, weights0))
    mini.append(min(arr))
    maxi.append(max(arr))
        


plt.plot(ran, vals, alpha=0.5, label="W")
plt.fill_between(ran, mini, maxi, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
    linewidth=4, linestyle='dashdot', antialiased=True)
plt.legend(loc='upper right')
plt.show(block = True)

'''
nsw_randomized = pd.read_csv("Data/nsw_randomized.txt")
nsw_observational = pd.read_csv("Data/nsw_observational.txt")

covariates = ["age","educ","black","hisp","marr","nodegree","re74","re75"]


nswr0, nswr1 = di.practical_counterfactual_distributions('treat',covariates,'re78',nsw_randomized, norm = False)
nswo0, nswo1 = di.practical_counterfactual_distributions('treat',covariates,'re78',nsw_observational, norm = False)

nswr = wasserstein_distance(nswr1.tolist(), nswr0.tolist())
nswo = wasserstein_distance(nswo1.tolist(), nswo0.tolist())
print(f'Randomized: {nswr}')
print(f'Observational: {nswo}')

x = np.linspace(0,1,100)
#pylab.plot(x,kernel1(x),"r", label = "Y(a')") # distribution function
#pylab.plot(x,kernel0(x),"b", label = 'Y(a)') # distribution function
pylab.hist(nswo1,density=1,alpha=.3, color = "m", label = "Y(a')") # histogram
pylab.hist(nswo0,density=1,alpha=.3, color = "g", label = "Y(a)") # histogram
pylab.legend(loc='upper right')
pylab.title("Counterfactual Densities")
pylab.show()


#plt.hist(a0, alpha=0.5, label="Y(a)")
#plt.hist(a1, alpha=0.5, label="Y(a')")
#plt.legend(loc='upper right')
#plt.show(block = True)
'''