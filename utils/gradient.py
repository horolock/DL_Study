import numpy as np

def numerical_diff(f, x):
    h = 1e-4                # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

############################
# Gradient
# A vector of partial differential of all variables
# f : function
# x : numpy array
############################
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)         # Create array that shape is same as 'x'

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)         # f(x + h)

        x[idx] = tmp_val - h
        fxh2 = f(x)         # f(x - h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val                # recover value
        it.iternext()
    
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
