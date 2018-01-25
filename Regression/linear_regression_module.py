import matplotlib
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


x = np.array([1,5,2,8,6,4], dtype=np.float64)
y = np.array([2,4,8,9,6,5], dtype=np.float64)

def ols(x,y):

    m = (((mean(x)*mean(y)) - mean(x*y)) /
         ((mean(x)*mean(x)) - mean(x*x)))

    b = mean(y) - m * mean(x)

    regression_line = [(m*x) + b for x in x]

    return m, b

ols(x,y)

m, b = ols(x,y)
regression_line = [(m * x) + b for i in x]

def squared_error(y_orig, y_estimated):
    return sum((y_orig - y_estimated) * (y_orig - y_estimated))

def r_squared(y_orig, y_estimated):
    y_mean_line = [mean(y_orig) for y in y_orig]
    squared_error_fitted = squared_error(y_orig, y_estimated)
    squared_error_mean = squared_error(y_orig, y_mean_line)
    return 1 - (squared_error_fitted/squared_error_mean)

r = r_squared(y, regression_line)
print(r)

plt.figure()
plt.scatter(x,y)
plt.scatter(x,y, color = "blue")
plt.plot(x, regression_line, color = "red")
plt.show()
