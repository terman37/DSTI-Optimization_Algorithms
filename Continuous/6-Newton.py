from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import math


def f(x):
    res = x[0] * math.exp(-(x[0] ** 2 + x[1] ** 2))
    # The rosenbrock function
    # res = 0
    # for i in range(x.shape[0] - 1):
    #     res += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return res


def df(x):
    dx = math.exp(-(x[0] ** 2 + x[1] ** 2)) - 2 * x[0] ** 2 * math.exp(-(x[0] ** 2 + x[1] ** 2))
    dy = -2 * x[1] * math.exp(-(x[0] ** 2 + x[1] ** 2))
    res = np.array([dx,dy])
    return res


def track_solution(xk):
    global costs
    costs.append(f(xk))


# initialization of costs followup
global costs
costs = []

# number of dimensions
D = 2
# bounds
x_min = -5 * np.ones(D)
x_max = 5 * np.ones(D)
bounds = (x_min, x_max)
cur_x = np.random.uniform(min(x_min), max(x_max), (1, D))

options = {'maxiter': 1000, 'xtol': 1e-3, 'disp': False}

result = optimize.minimize(f, x0=cur_x, method="Newton-CG", jac=df, options=options, callback=track_solution)

print("--- Newton minimization ---")
print("%s in %d iterations" % (result.message, result.nit))
print("Best solution found: %f at point:" % result.fun, result.x)

plt.plot(costs)
plt.show()
