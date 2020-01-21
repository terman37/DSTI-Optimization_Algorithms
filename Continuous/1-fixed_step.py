import matplotlib.pyplot as plt
import numpy as np


def f(v):
    res = 0
    for x in v:
        res += x ** 2 + 5
    return res


def df(v):
    res = 0
    res += 2 * v
    return res


debug = True

# number of dimensions
D = 2
# bounds
x_min = -5 * np.ones(D)
x_max = 5 * np.ones(D)
bounds = (x_min, x_max)
# Starting point
cur_x = np.random.uniform(min(x_min), max(x_max), (1, D))
cur_fx = f(cur_x[-1])
# Algorithm parameters
# - learning Rate --> ro if df lipschitz continuous 0 < ro < 2alpha/m² --> to define
rate = 0.01
# - Precision (when to stop the algorithm)
# precision = 1e-4
# - maximum number of iterations
max_iters = 100
# - initial step size
# previous_step_size = 1

# counter
iters = 0

# while previous_step_size > precision: and iters < max_iters:
while iters < max_iters:
    prev_x = cur_x[-1]
    prev_fx = f(cur_x[-1])
    grad_desc = cur_x[-1] - rate * df(prev_x)
    cur_x = np.vstack((cur_x, grad_desc))
    cur_fx = np.vstack((cur_fx, f(cur_x[-1])))
    # previous_step_size = np.linalg.norm(cur_x[-1] - prev_x)
    # previous_step_size -= rate
    iters += 1
    if debug:
        print("Iteration", iters, " X value is", cur_x[-1])  # Print iterations

print("Best solution:", f(cur_x[-1]))
print("at point:", cur_x[-1])
print("found in %d iterations" % iters)

# x = np.arange(min(x_min), max(x_max), .1)
# # fx = [f(i) for i in x]
# optx = [f(i) for i in cur_x]

plt.plot(cur_fx)
# plt.plot(cur_x, optx, "o")
plt.show()
