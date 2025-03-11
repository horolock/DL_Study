import numpy as np

from common import im2col, col2im

class Pooling:
    def __init__(self, poolH, poolW, stride=2, pad=0):
        self.poolH = poolH
        self.poolW = poolW
        self.stride = stride
        self.pad = pad

        self.x = None
        self.argMax = None

    def forward(self, x):
        N, C, H, W = x.shape

        outH = int(1 + (H - self.poolH) / self.stride)
        outW = int(1 + (W - self.poolW) / self.stride)

        # 전개
        col = im2col(x, self.poolH, self.poolW, self.stride, self.pad)
        col = col.reshape(-1, self.poolH * self.poolW)

        # 최댓값
        argMax = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        # Reshape
        out = out.reshape(N, outH, outW, C).transpose(0, 3, 1, 2)

        self.x = x
        self.argMax = argMax

        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        poolSize = self.poolH * self.poolW

        dMax = np.zeros((dout.size, poolSize))
        dMax[np.arange(self.argMax.size), self.argMax.flatten()] = dout.flatten()
        dMax = dMax.reshape(dout.shape + (poolSize,))

        dcol = dMax.reshape(dMax.shape[0] * dMax.shape[1] * dMax.shape[2], -1)

        dx = col2im(dcol, self.x.shape, self.poolH, self.poolW, self.stride, self.pad)

        return dx