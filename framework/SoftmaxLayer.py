import numpy as np
from framework.Layer import Layer

class SoftmaxLayer(Layer):
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        shifted = dataIn - np.max(dataIn, axis=1, keepdims=True)
        exps = np.exp(shifted)
        output = exps / np.sum(exps, axis=1, keepdims=True)
        self.setPrevOut(output)
        return output

    def gradient(self):
        po = self.getPrevOut()  # shape: (batch, num_classes)
        batch_size, dims = po.shape
        grad = np.zeros((batch_size, dims, dims))
        for i in range(batch_size):
            p = po[i, :]
            grad[i, :, :] = np.diag(p) - np.outer(p, p)
        return grad

    def backward(self, gradIn):
        return np.einsum('nij,ni->nj', self.gradient(), gradIn)
