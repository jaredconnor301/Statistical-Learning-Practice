import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
style.use( 'fivethirtyeight')

# Define a dataset: dictionary of 2 variables, 3 data points of 2d coordinates
data = {'b':[[1,1],[2,1],[1,3]], 'r':[[5,4],[7,5],[8,7]]}
new_feature = [6,6]

# Plotting the exisitng points
for i in data:
    for ii in data[i]:
        plt.scatter(ii[0],ii[1],s=100, color = i)


# define k nearest neighbors function
def k_nearest_neighbors(data, predict, k = 3):

    if len(data) > k:
        warnings("The dataset is larger than the K parameter. Increase the size of K")


    return vote_result
