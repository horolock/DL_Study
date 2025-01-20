import sys, os
sys.path.append(os.pardir)

import numpy as np
from utils.functions import softmax, cross_entropy_error
from utils.gradient import numerical_gradient

