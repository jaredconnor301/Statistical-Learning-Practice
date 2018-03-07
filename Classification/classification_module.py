import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)

df.head()

# Define our X & y for labels and features respectively
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# Define the cross_validation.train_test_split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
test_size = 0.1)

# Define the classified: clf
clf = neighbors.KNeighborsClassifier()

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Print the accuracy with clf.score()
accuracy = clf.score(X_test, y_test)
print(accuracy)

# testing additioanl values to view the effect on prediction
example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print(prediction)

# testing two predictions on the same classifier
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(2,-1)
prediction = clf.predict(example_measures)
print(prediction)
