import numpy as np

def im2col(inputData, filterH, filterW, stride=1, pad=0):
    # image -> 2D Flatten
    N, C, H, W = inputData.shape

    outH = (H + (2 * pad) - filterH) // stride + 1
    outW = (W + (2 * pad) - filterW) // stride + 1

    img = np.pad(inputData, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
    col = np.zeros((N, C, filterH, filterW, outH, outW))

    for y in range(filterH):
        yMax = y + stride * outH

        for x in range(filterW):
            xMax = x + stride * outH

            col[:, :, y, x, :, :] = img[:, :, y:yMax:stride, x:xMax:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * outH * outW, -1)

    return col

def col2im(col, inputShape, filterH, filterW, stride=1, pad=0):
    # 2D -> Images 

    N, C, H, W = inputShape

    outH = (H + (2 * pad) - filterH) // stride + 1
    outW = (W + (2 * pad) - filterW) // stride + 1

    col = col.reshape(N, outH, outW, C, filterH, filterW).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + (2 * pad) + stride - 1, W + (2 * pad) + stride - 1))

    for y in range(filterH):
        yMax = y + stride * outH

        for x in range(filterW):
            xMax = x + stride * outW

            img[:, :, y:yMax:stride, x:xMax:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

