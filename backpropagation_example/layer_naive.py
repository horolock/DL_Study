import numpy as np

# Multiply layer
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # Forward propagation
    def forward(self, x, y):
        self.x = x
        self.y = y

        out = x * y

        return out
    
    # Backward propagation
    def backward(self, dout):
        #####################################################################
        # In here, `self.x` and `self.y` meaning values that after forward.
        # `dout` value that result of upper differential
        #####################################################################
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


# Add layer
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
