import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
import pandas as pd
import random
style.use( 'fivethirtyeight')

# define k nearest neighbors function
def k_nearest_neighbors(data, predict, k = 3):

    if len(data) > k:
        warnings("The dataset is larger than the K parameter. Increase the size of K")

    distance = []

    for group in data:
        for feature in data[group]:
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distance.append([euclidean_distance, group])

    for i in sorted(distance)[:k]:
        votes = i[1]

    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

# Read in the breast cancer dataset
cancer_data = pd.read_csv("/Users/Jared/Documents/Programming/Python/Statistical Learning/Classification/breast-cancer-wisconsin.data.txt")
cancer_data.replace(to_replace="?", value=-99999, inplace=True)
cancer_data.head()
cancer_data.drop(["id"], 1, inplace=True)
