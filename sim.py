import sim_functions as sf
import pandas as pd
import random


# Set the sample size
nums = 1000

# Create the DAG using the nonparametric structural equations
data = pd.DataFrame()
data['A'] = sf.generate_normal(1, 2, nums)
data['B'] = sf.generate_uniform(0, 1, nums)
data['D'] = data['B']/3 + sf.generate_gauss(0, 1, nums)
data['C'] = data['B'] + data['A'] - sf.generate_uniform(0, 1, nums)
# Does not end up being binary, create a vector of probs to figure out what T will be 
data['T'] = sf.generate_treatment(data['B'], 0.5, 0.1)
data['F'] = data['T']*2 + sf.generate_gauss(0, 1, nums)
data['Y'] = data['A'] + data['C'] + data['D'] + data['F'] + sf.generate_uniform(4, 2, nums)

# Compare the ACE when conditioning on A,C,D and when conditioning on B
#print('ACE when conditioning on A, C, and D:')
#sf.print_ace('Y','T',['A','C','D'], data)
#print('ACE when conditioning on B:')
#sf.print_ace('Y','T',['B'], data)

# Compute the log likelihood ratio of some of the dependencies
print('Conditional Dependencies: Value below 0.05 implies dependence')
print('Dependent:')
print('B ~ T | F')
sf.print_likelihood('B','T',['F'],data)


print('\n\nIndependent:')
print('B ~ F | D and T')
sf.print_likelihood('B','F',['D','T'], data)
print('D ~ C | B')
sf.print_likelihood('C','D',['B'],data)
sf.print_likelihood('A','B',['D'],data)
sf.print_likelihood('A','B',['T'],data)
sf.print_likelihood('A','B',['F'],data)
sf.print_likelihood('A','D',['B'],data)
sf.print_likelihood('A','D',['T'],data)
sf.print_likelihood('A','D',['F'],data)

#sf.counter_plot('Y','T',['A','C','D'], data)






