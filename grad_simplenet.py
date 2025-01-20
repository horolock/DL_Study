import sys, os
sys.path.append(os.pardir)

import numpy as np
from utils.functions import softmax, cross_entropy_error
from utils.gradient import numerical_gradient

# SimpleNet
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)
    
    ##############
    # Loss function's output
    #  x    :   Input data
    #   t   :   Answer Label
    ##############
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
    

net = simpleNet()
print(f"W   :   {net.W}")

x = np.array([0.6, 0.9])
p = net.predict(x)

print(f"p   :   {p}")

t = np.array([0, 0, 1])
print(f"Net Loss output : {net.loss(x, t)}")

def f(W):
    return net.loss(x, t)
# f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)

print(dW)