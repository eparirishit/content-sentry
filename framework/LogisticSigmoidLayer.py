import numpy as np
from .Layer import Layer

class LogisticSigmoidLayer(Layer):
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        output = 1 / (1 + np.exp(-dataIn))
        self.setPrevOut(output)
        return output

    def gradient(self):
        return self.getPrevOut() * (1 - self.getPrevOut())

    def backward(self, gradIn):
        return gradIn * self.gradient()