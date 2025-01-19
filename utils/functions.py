import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int8)

#############################
# Sigmoid activation function
#############################
def sigmoid(x):
    #       1
    #   ------------
    #   1 + exp(-x)
    return 1 / (1 + np.exp(-x))

###########################
# Relu activation function
###########################
def relu(x):
    return np.maximum(0, x)

###############
# Output Layer
###############
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)               # Get each element's exp(a), (a - c) => Preventing overflow
    sum_exp_a = np.sum(exp_a)           # Sum every elements 
    y = exp_a / sum_exp_a
    return y

def identity_function(x):
    return x

################################
# Sum of Squares of Error (SSE)
# 
# E = 1/2 * sum((yk - tk)^2)
# y : Neural network's output
# t : Answer label
################################
def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t)**2)

##############################
#   Cross Entropy Error (CEE)
#
#   E = -sum(tk * log(yk))
#   y : Neural network output
#   t : Answer label
##############################
def cross_entropy_error(y, t):
    delta = 1e-7        # if np.log()'s input is 0, it can be -inf so plus delta(really small value) to y to prevent -inf
    return -np.sum(t * np.log(y + delta))
