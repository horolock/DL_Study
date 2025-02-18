import sys, os
sys.path.append(os.pardir)

import numpy as np

from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialize weights
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # Create Layer
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    # Process predict
    # x - Image data
    def predict(self, x):
        # Do something...
        pass
    
    # Get loss function's value
    # x - Image data
    # t - Answer label
    def loss(self, x, t):
        pass

    # Get accuracy
    def accuracy(self, x, t):
        pass

    # Get slope with backpropagation
    def gradient(self, x, t):
        pass
