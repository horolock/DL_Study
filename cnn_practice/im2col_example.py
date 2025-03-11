import sys, os
sys.path.append(os.pardir)

import numpy as np
from common import im2col

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, 1, 0)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, 1, 0)
print(col2.shape)