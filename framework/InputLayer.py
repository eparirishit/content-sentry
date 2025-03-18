import numpy as np
from framework.Layer import Layer

class InputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0, keepdims=True)
        self.stdX = np.std(dataIn, axis=0, keepdims=True, ddof=1)
        self.stdX[self.stdX == 0] = 1

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        zscored = (dataIn - self.meanX) / self.stdX
        self.setPrevOut(zscored)
        return zscored

    def gradient(self):
        return None
