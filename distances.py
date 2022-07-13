from hashlib import new
import scipy.stats as stats
from sklearn import metrics
import numpy as np
import density_ipw as di
import pandas as pd
import sim_functions as sf
import random
import matplotlib.pyplot as plt


def wass(x, y):
    return stats.wasserstein_distance(x.tolist(), y.tolist())

def mmd(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    XX = metrics.pairwise.rbf_kernel(x, x, 2)
    XY = metrics.pairwise.rbf_kernel(x, y, 2)
    YY = metrics.pairwise.rbf_kernel(y, y, 2)
    return XX.mean() + YY.mean() - 2*XY.mean()

def l2(x, y):
    return None

def hell(x, y):
    return None

def chi(x, y):
    return None

def ks(x, y):
    return stats.ks_2samp(x, y).statistic

def test_distance(A, C, Y, data, distance, num_bootstraps = 10000):
    x, y = di.theoretical_counterfactual_distributions(A, 'theor', Y, data)
    dist = distance(x, y)
    combined = np.append(x, y)
    arr = []
    count = 0
    for i in range(num_bootstraps):
        new_data_x = np.random.choice(combined, len(x))
        new_data_y = np.random.choice(combined, len(y))
        dist2 = distance(new_data_x, new_data_y)
        arr.append(dist2)
        if dist2 >= dist:
            count += 1

    pvalue = count / num_bootstraps if count > 0 else 1 / num_bootstraps
    return dist, pvalue

def run_tests():
    nums = 1000
    size = 10
    distances = [wass, mmd, ks]
    names = ['Wasserstein','MMD', 'Kolmogorov-Smirnov']
    results = [[0]*size for i in range(len(distances))]
    ran = [i/20 for i in range(size)]
    for k in range(size):
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
        data['Y'] = data['C'] + (k/20)*data['A'] + sf.generate_uniform(-1, 1, nums)

        for j, distance in enumerate(distances):
            stat, pvalue = test_distance('A',['C'], 'Y', data, distance)
            stat2, pvalue2 = test_distance('A',['C'], 'Y', data, distance)
            print(distance, stat, stat2)
            results[j][k] = pvalue
    for i, result in enumerate(results):
        plt.plot(ran,result, alpha=0.5, label=names[i])
    plt.legend(loc='upper right')
    plt.show(block = True)


def main():
    run_tests()


if __name__ == "__main__":
    main()

