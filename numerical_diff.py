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


######################
# Partial differential (편미분) Example
######################
def function_2(x):
    # f(x0, x1) = x0^2 + x1^2
    # Or return np.sum(x**2)
    return x[0]**2 + x[1]**2

############################
# Gradient
# A vector of partial differential of all variables
# f : function
# x : numpy array
############################
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)         # Create array that shape is same as 'x'

    for idx in range(x.size):
        tmp_val = x[idx]

        # Calculate f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # Calculate f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val 
    
    return grad

##################
# Gradient Descent  (경사 하강법)
#   f                : function to optimize
#   init_x        : initialize value
#   lr              : Learning Rate
#   step_num : repeat count
#########################
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x

# Normal case
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

# if learning rate is too big
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))

# if learning rate is too small
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))