import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int8)

# numpy의 활용
def test_1():
    x = np.array([-1.0, 1.0, 2.0])
    print(x)                    # [-1.  1.  2.]

    y = x > 0
    print(y)                    # [False  True  True]
    print(y.astype(np.int8))    # [0 1 1]

test_1()

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)         # Declare y range
plt.show()


#############################
# Sigmoid activation function
#############################
def sigmoid(x):
    #       1
    #   ------------
    #   1 + exp(-x)
    return 1 / (1 + np.exp(-x))

# Draw sigmoid function as graph
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)         # Declare y range
plt.show()

###########################
# Relu activation function
###########################
def relu(x):
    return np.maximum(0, x)