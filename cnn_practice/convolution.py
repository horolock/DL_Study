import numpy as np

from common import im2col, col2im

class Convolution:
    def __init__(self, w, b, stride=1, pad=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

        # Middle Data (for backward)
        self.x = None
        self.col = None
        self.colW = None

        # Weight, bias gradient
        self.dw = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.w.shape
        N, C, H, W = x.shape

        outH = 1 + int((H + 2 * self.pad - FH) / self.stride)
        outW = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        colW = self.w.reshape(FN, -1).T

        out = np.dot(col, colW) + self.b
        out = out.reshape(N, outH, outW, -1).transpose(0, 3, 1, 2)  # N, C, H, W

        self.x = x
        self.col = col
        self.colW = colW

        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dw = np.dot(self.col.T, dout)
        self.dw = self.dw.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.colW.T)

        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx



