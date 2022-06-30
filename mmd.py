import random
import time
import math
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy
import torch


def gaussian_kernel(x, y, sigma = 0.5):
    form = -(x**2 + y**2)/(2*(sigma**2))
    return math.sqrt(math.exp(form)**2)

#
# Compute the MMD of two sequences of data
#
def mmd(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    XX = metrics.pairwise.rbf_kernel(x, x, 2)
    XY = metrics.pairwise.rbf_kernel(x, y, 2)
    YY = metrics.pairwise.rbf_kernel(y, y, 2)
    return XX.mean() + YY.mean() - 2*XY.mean()

#
# Compute the MDD and p-value based on combined distribution sample
#
def ci_mmd(x, y, plot = False, num_bootstraps=1000, alpha=0.05):
    n = len(x)
    combined = np.append(x, y)
    ker_comb = combined.reshape(-1, 1)

    dists = torch.pdist(torch.from_numpy(combined)[:,None])
    sigma = dists[:100].median()/2
    ker = metrics.pairwise.rbf_kernel(ker_comb, ker_comb, 1/2*(sigma.numpy()**2))
    
    xx = ker[:n, :n]
    yy = ker[n:, n:]
    xy =ker[:n, n:]
    # Find MMD
    mmd = xx.mean() + yy.mean() - 2*xy.mean()

    m = len(combined)
    arr = []
    count = 0
    # Bootstrap
    for i in range(num_bootstraps):
        new_data = [random.randrange(m) for i in range(m)]
        #new_y = random.sample(range(0, m - 1), m - n)
        new_ker = ker[np.ix_(new_data, new_data)]
        new_xx = new_ker[:n, :n]
        new_yy = new_ker[n:, n:]
        new_xy = new_ker[:n, n:]
        new_mmd = new_xx.mean() + new_yy.mean() - 2*new_xy.mean()
        arr.append(new_mmd)
        if new_mmd >= mmd:
            count += 1
    
    if plot:
        plt.hist(arr, alpha=0.5, label="CI")
        plt.legend(loc='upper right')
        plt.show(block = True)

    pvalue = count / num_bootstraps if count > 0 else 1 / num_bootstraps
    return mmd, pvalue
