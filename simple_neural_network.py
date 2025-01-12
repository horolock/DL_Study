import numpy as np

X = np.array([1, 2])            # input x1, x2
print(X.shape)

W = np.array([
    [1, 3, 5],
    [2, 4, 6]
])                              # Weights 

print(W.shape)

Y = np.dot(X, W)                # Output y1, y2, y3
print(Y)

################################
# Activation function : Sigmoid
################################
def sigmoid(x):
    #       1
    #   ------------
    #   1 + exp(-x)
    return 1 / (1 + np.exp(-x))

############################################
# Simple Neural Network implementation Start
############################################
X = np.array([1.0, 0.5])
W1 = np.array([
    [0.1, 0.3, 0.5],
    [0.2, 0.4, 0.6]
])
B1 = np.array([0.1, 0.2, 0.3])

# Layer 1
A1 = np.dot(X, W1) + B1
print(f"A1 : {A1}")

Z1 = sigmoid(A1)                    # a --h()--> z
print(f"Z1 : {Z1}")

# Layer 2
W2 = ([
    [0.1, 0.4],
    [0.2, 0.5],
    [0.3, 0.6]
])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
print(f"A2 : {A2}")
Z2 = sigmoid(A2)
print(f"Z2 : {Z2}")

# Output Layer
def identity_function(x):
    # Output activation function
    return x

W3 = np.array([
    [0.1, 0.3],
    [0.2, 0.4]
])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
print(f"A3 : {A3}")
Y = identity_function(A3)
print(f"Y : {Y}")
############################################
# Simple Neural Network implementation End
############################################


###########
# Clean up
###########
def create_network():
    network = {}
    network['W1'] = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6]
    ])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6]
    ])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([
        [0.1, 0.3],
        [0.2, 0.4]
    ])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    y = identity_function(a3)

    return y

network = create_network()
inp = ([1.0, 0.5])
out = forward(network, inp)
print(out)