import numpy as np
from framework.Layer import Layer

class ReLULayer(Layer):
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        output = np.maximum(0, dataIn)
        self.setPrevOut(output)
        return output

    def gradient(self):
        X = self.getPrevIn()
        return (X > 0).astype(float)

    def backward(self, gradIn):
        derivative = self.gradient()
        return gradIn * derivative
