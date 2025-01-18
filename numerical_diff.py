import numpy as np
import matplotlib.pylab as plt

##################
# First numerical diff function
# f : function
# x : parameter to give function f
##################
def numerical_diff_first(f, x):
    h = 1e-50                               # This can cause rounding error
    return (f(x + h) - f(x)) / h    # f(x + h) - f(x) include the number of error (오차가 발생함)

def numerical_diff(f, x):
    h = 1e-4                # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

##########
# Example Code
##########
def function_1(x):
    # 0.01x^2 + 0.1x
    return 0.01*(x**2) + 0.1*x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)           # 0.0 ~ 20.0 with 0.1 step
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)                                      # tf (tangent_line) returns lambda

tf = tangent_line(function_1, 10)
y3 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)

plt.show()
