import numpy as np
from framework.Layer import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        self.weights = np.random.uniform(-1e-4, 1e-4, (sizeIn, sizeOut))
        self.biases = np.random.uniform(-1e-4, 1e-4, (1, sizeOut))
        self.gradW = None
        self.gradB = None

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

    def getBiases(self):
        return self.biases

    def setBiases(self, biases):
        self.biases = biases

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        output = np.dot(dataIn, self.weights) + self.biases
        self.setPrevOut(output)
        return output

    def gradient(self):
        batch_size = self.getPrevIn().shape[0]
        return np.array([self.weights.T for _ in range(batch_size)])

    def backward(self, gradIn):
        X = self.getPrevIn()  # (batch, sizeIn)
        self.gradW = np.dot(X.T, gradIn)  # (sizeIn, sizeOut)
        self.gradB = np.sum(gradIn, axis=0, keepdims=True)  # (1, sizeOut)
        return np.dot(gradIn, self.weights.T)  # (batch, sizeIn)

    def updateWeights(self, eta):
        """Updated method with correct signature"""
        if self.gradW is not None and self.gradB is not None:
            self.weights -= eta * self.gradW / self.gradW.shape[0]
            self.biases -= eta * self.gradB / self.gradB.shape[0]
            self.gradW = None
            self.gradB = None
