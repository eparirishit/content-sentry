import numpy as np
from framework.Layer import Layer

class TanhLayer(Layer):
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        output = np.tanh(dataIn)
        self.setPrevOut(output)
        return output

    def gradient(self):
        out = self.getPrevOut()
        return 1 - out ** 2

    def backward(self, gradIn):
        return gradIn * self.gradient()
