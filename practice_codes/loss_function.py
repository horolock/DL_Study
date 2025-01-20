import numpy as np

################################
# Sum of Squares of Error (SSE)
# 
# E = 1/2 * sum((yk - tk)^2)
# y : Neural network's output
# t : Answer label
################################
def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t)**2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]                              # The answer is '2'
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]        # Neural network's output is predict '2' as 60%
print(f"Nerual network think it's 2 (but answer is 2) : {sum_squares_error(np.array(y), np.array(t))}")

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]        # This time, Neural network's output is predict '7' as 60%
print(f"Nerual network think it's 7 (but answer is 2) : {sum_squares_error(np.array(y), np.array(t))}")


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

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]        # Neural network's output is predict '2' as 60%
print(f"Nerual network think it's 2 (but answer is 2) : {cross_entropy_error(np.array(y), np.array(t))}")

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]        # This time, Neural network's output is predict '7' as 60%
print(f"Nerual network think it's 7 (but answer is 2) : {cross_entropy_error(np.array(y), np.array(t))}")

