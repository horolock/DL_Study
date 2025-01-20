# Create Logic Gate AND, NAND, OR with Perceptron

import numpy as np

def AND(x1, x2):
    # Weights and Threshold
    w1, w2, theta = 0.5, 0.5, 0.7
    result = (x1 * w1) + (x2 * w2)

    if result <= theta:
        return 0
    elif result > theta:
        return 1
    
    return 0

print('#####    AND Gate    #####')
print(AND(0,0))     # 0
print(AND(1,0))     # 0
print(AND(0,1))     # 0
print(AND(1,1))     # 1
print('##########################')

# Using Numpy
def NumpyAND(x1, x2):
    x = np.array([x1, x2])      # input
    w = np.array([0.5, 0.5])    # weight
    b = -0.7                    # bias

    res = np.sum(x * w) + b

    if res <= 0:
        return 0
    elif res > 0:
        return 1
    
    return 0

print('#####    Numpy AND Gate    #####')
print(NumpyAND(0,0))     # 0
print(NumpyAND(1,0))     # 0
print(NumpyAND(0,1))     # 0
print(NumpyAND(1,1))     # 1
print('################################')

def NAND(x1, x2):
    x = np.array([x1, x2])      # input
    w = np.array([-0.5, -0.5])  # weights
    b = 0.7                     # bias

    res = np.sum(x * w) + b

    if res <= 0:
        return 0
    elif res > 0:
        return 1
    return 0

print('#####    NAND Gate    #####')
print(NAND(0,0))
print(NAND(1,0))
print(NAND(0,1))
print(NAND(1,1))
print('###########################')

def OR(x1, x2):
    x = np.array([x1, x2])              # input
    w = np.array([1.0, 1.0])            # weights
    b = -0.5                            # bias

    res = np.sum(x * w) + b

    if res <= 0:
        return 0
    elif res > 0:
        return 1
    
    return 0

print('#####    OR Gate    #####')
print(OR(0,0))
print(OR(1,0))
print(OR(0,1))
print(OR(1,1))
print('#########################')

# XOR Gate
# This is multi-layer perceptron
def XOR(x1, x2):
    nand_res = NAND(x1, x2)
    or_res = OR(x1, x2)
    and_res = AND(nand_res, or_res)

    return and_res

print('#####    XOR Gate    #####')
print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
print('##########################')
    
