# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:26:15 2018

@author: JaredConnor
"""
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("breast-cancer-wisconsin.data.txt", error_bad_lines=False)
df.replace("?", -99999, inplace = True)
df.drop(['id'], 1, inplace = True)

# Define our features and labels 
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# Set the test/train split at 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20)

# Specify the classified
clf = neighbors.KNeighborsClassifier()

# Fit the mdoel
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)