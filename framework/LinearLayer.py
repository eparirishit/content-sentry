import numpy as np
from framework.Layer import Layer

class LinearLayer(Layer):
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn

    def gradient(self):
        return np.array([np.eye(self.getPrevIn().shape[1]) for _ in range(self.getPrevIn().shape[0])])

    def backward(self, gradIn):
        return gradIn
