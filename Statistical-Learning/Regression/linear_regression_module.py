from statistics import mean
import statistics
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(3)


def generate_dataset(n, variance, step = 2, correlation = False):
    val = 1
    yi = []
    for i in range(n):
        y = val + random.randrange(-variance,variance)
        yi.append(y)
        if correlation =='pos':
            val += step
        elif correlation == 'neg':
            val -= step
            
    xi = [i for i in range(len(yi))]
    
    return np.array(xi, dtype=np.float64), np.array(yi, dtype=np.float64)

def ols(xi,yi):

    m = (((mean(xi)*mean(yi)) - mean(xi*yi)) /
         ((mean(xi)*mean(xi)) - mean(xi*xi)))

    b = mean(yi) - m * mean(xi)

    return m, b


def squared_error(y_orig, y_estimated):
    return sum((y_orig - y_estimated) * (y_orig - y_estimated))


def r_squared(y_orig, y_estimated):
    y_mean_line = [mean(y_orig) for y in y_orig]
    squared_error_fitted = squared_error(y_orig, y_estimated)
    squared_error_mean = squared_error(y_orig, y_mean_line)
    return 1 - (squared_error_fitted/squared_error_mean)


xi, yi = generate_dataset(100, 100, 2, correlation = 'neg')
m, b = ols(xi,yi)
regression_line = [(m * x) + b for x in xi]
r = r_squared(yi, regression_line)
print(r)

plt.figure()
plt.scatter(xi,yi)
plt.scatter(xi,yi, color = "blue")
plt.plot(xi, regression_line, color = "red")
plt.show()
