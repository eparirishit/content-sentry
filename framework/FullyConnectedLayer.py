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

    def updateWeightsWithMomentum(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, momentum=None, velocity=None):
        """Update weights using Adam-like optimization with momentum and adaptive learning rates"""
        # Check if gradients exist
        if self.gradW is None or self.gradB is None:
            return
            
        # Initialize momentum and velocity for this layer if not already in the dictionaries
        layer_id = id(self)
        
        if momentum is None:
            momentum = {}
        if velocity is None:
            velocity = {}
            
        if layer_id not in momentum:
            momentum[layer_id] = {
                'W': np.zeros_like(self.weights),
                'b': np.zeros_like(self.biases)
            }
            velocity[layer_id] = {
                'W': np.zeros_like(self.weights),
                'b': np.zeros_like(self.biases)
            }
        
        # Update momentum with current gradients
        momentum[layer_id]['W'] = beta1 * momentum[layer_id]['W'] + (1 - beta1) * self.gradW
        momentum[layer_id]['b'] = beta1 * momentum[layer_id]['b'] + (1 - beta1) * self.gradB
        
        # Update velocity with squared gradients
        velocity[layer_id]['W'] = beta2 * velocity[layer_id]['W'] + (1 - beta2) * np.square(self.gradW)
        velocity[layer_id]['b'] = beta2 * velocity[layer_id]['b'] + (1 - beta2) * np.square(self.gradB)
        
        # Calculate adaptive learning rates
        adaptive_lr_W = learning_rate / (np.sqrt(velocity[layer_id]['W']) + epsilon)
        adaptive_lr_b = learning_rate / (np.sqrt(velocity[layer_id]['b']) + epsilon)
        
        # Update weights and biases
        self.weights -= adaptive_lr_W * momentum[layer_id]['W']
        self.biases -= adaptive_lr_b * momentum[layer_id]['b']
        
        # Reset gradients after update
        self.gradW = None
        self.gradB = None
