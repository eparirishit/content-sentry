import numpy as np
from framework.Layer import Layer

class TransposeLayer(Layer):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        output = np.transpose(dataIn, self.axes)
        self.setPrevOut(output)
        return output

    def backward(self, gradIn):
        return np.transpose(gradIn, np.argsort(self.axes))

    def gradient(self):
        return None