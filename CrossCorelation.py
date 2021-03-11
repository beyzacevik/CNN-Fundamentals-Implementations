import torch
import numpy as np

class CrossCorrelation2D(object):

    def __init__(self, X, K):
        self.x = X
        self.k = K
        self.n_h = X.shape[0]
        self.n_w = X.shape[1]
        self.k_h = K.shape[0]
        self.k_w = K.shape[1]

    def cross_correlation_2d(self):

        y = torch.zeros((self.n_h - self.k_h + 1), (self.n_w - self.k_w + 1))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i, j] = reduced_sum(self.x[i:i+self.k_h, j:j+self.k_w] * self.k)

        return y

def reduced_sum(matrix):
    total = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            total += matrix[i, j]
    return total


X = torch.from_numpy(np.arange(9).reshape(3, 3))
K = torch.from_numpy(np.arange(4).reshape(2, 2))
print(X)
print(K)
conv2d = CrossCorrelation2D(X, K)
print(conv2d.cross_correlation_2d())