from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    res = 0
    for i in range(x.shape[0] - 1):
        res += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return res

def track_solution(xk):
    global costs
    costs.append(f(xk))


# initialization of costs followup
global costs
costs = []

# number of dimensions
D = 50
# bounds
x_min = -5 * np.ones(D)
x_max = 5 * np.ones(D)
bounds = (x_min, x_max)
cur_x = np.random.uniform(min(x_min), max(x_max), (1, D))

options = {'maxiter': 1000, 'gtol': 1e-3, 'disp': False}

# Minimization of scalar function of one or more variables using CG / BFGS algorithm.
result = optimize.minimize(f, x0=cur_x, method="BFGS", options=options, callback=track_solution)

print("--- Conjugate gradient minimization ---")
print("%s in %d iterations" % (result.message, result.nit))
print("Best solution found: %f at point:" % result.fun, result.x)

plt.plot(costs)
plt.show()
