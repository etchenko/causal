import matplotlib.pyplot as plt
import pylab
from scipy import stats
import numpy as np
import pandas as pd
import sim_functions as sf
import random


data = pd.DataFrame()
nums = 1000
data['a'] = sf.generate_gauss(0,1,nums)
data['b'] = sf.generate_gauss(0,1,nums)
for i in range(nums):
        data['a'][i] = np.random.normal(-1, 3, 1)[0] + np.random.laplace(0, 1, 1)[0]*3 - 2
        data['b'][i] = random.uniform(0, 2) - (np.random.laplace(1, 2,1)[0]*4) + 50

kernel2 = stats.gaussian_kde(data['a'], 'scott')
kernel3 = stats.gaussian_kde(data['b'], 'scott')

x = np.linspace(min(data['a']), max(data['a']),1000)
y = np.linspace(min(data['b']), max(data['b']), 1000)

nums = 10000

pylab.plot(x,kernel2(x),"b", label = "Treatment") # distribution function
pylab.plot(y,kernel3(y),"r", label = 'No Treatment') # distribution function
pylab.hist(data['a'],density=1,alpha=.3, color = "b") # histogram
pylab.hist(data['b'],density=1,alpha=.3, color = "r") # histogram
pylab.legend(fontsize=18, title_fontsize=20)
pylab.ylabel('p(Y | do(A))', fontsize = 15)
pylab.xlabel('Outcome', fontsize = 15)
#pylab.title("After IPW")
pylab.show()

data = pd.DataFrame()
nums = 1000
data['a'] = sf.generate_gauss(0,1,nums)
data['b'] = sf.generate_gauss(0,1,nums)
for i in range(nums):
        data['a'][i] = np.random.normal(0, 3, 1)[0] + np.random.laplace(0, 1, 1)[0]*3 + np.random.normal(0, 6, 1)[0]
        data['b'][i] = np.random.normal(-15, 5, 1)[0] if random.random() > 0.5 else np.random.normal(15, 5, 1)[0]

kernel2 = stats.gaussian_kde(data['a'], 'scott')
kernel3 = stats.gaussian_kde(data['b'], 'scott')

x = np.linspace(min(data['a']), max(data['a']),1000)
y = np.linspace(min(data['b']), max(data['b']), 1000)

nums = 10000

pylab.plot(x,kernel2(x),"b", label = "Treatment") # distribution function
pylab.plot(y,kernel3(y),"r", label = 'No Treatment') # distribution function
pylab.hist(data['a'],density=1,alpha=.3, color = "b") # histogram
pylab.hist(data['b'],density=1,alpha=.3, color = "r") # histogram
pylab.legend(fontsize=18, title_fontsize=20)
pylab.ylabel('p(Y | do(A))', fontsize = 15)
pylab.xlabel('Outcome', fontsize = 15)
#pylab.title("After IPW")
pylab.show()