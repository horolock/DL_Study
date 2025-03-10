import sys, os
sys.path.append(os.pardir)

import numpy as np

from dataset.mnist import load_mnist

# Load mnist dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

# Get random data
train_size = x_train.shape[0]
batch_size = 10

batch_mask = np.random.choice(train_size, batch_size)

print(batch_mask)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

###################
# Cross Entropy Error for Batch
###################
def cross_entropy_error(y, t):
    if y.ndim == 1:
        # If neural network's output data is one dimension (not a batch), reshape the data
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

#####################################
# Cross Entry Error for Batch
# This is for the case that answer label is not 'one-hot-encoding'
#####################################
def cross_entropy_error2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size