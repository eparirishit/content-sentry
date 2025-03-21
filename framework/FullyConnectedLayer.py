import numpy as np
from framework.Layer import Layer

class FullyConnectedLayer(Layer):
    """
    Fully connected (dense) layer implementation.
    
    Implements a traditional fully connected neural network layer with
    weights, biases, forward pass, and gradient computation.
    """
    def __init__(self, sizeIn, sizeOut):
        """
        Initialize a fully connected layer.
        
        Args:
            sizeIn: Number of input features
            sizeOut: Number of output features
        """
        super().__init__()
        self.weights = np.random.uniform(-1e-4, 1e-4, (sizeIn, sizeOut))
        self.biases = np.random.uniform(-1e-4, 1e-4, (1, sizeOut))
        self.gradW = None
        self.gradB = None

    def getWeights(self):
        """Get the current weights."""
        return self.weights

    def setWeights(self, weights):
        """Set weights to specified values."""
        self.weights = weights

    def getBiases(self):
        """Get the current biases."""
        return self.biases

    def setBiases(self, biases):
        """Set biases to specified values."""
        self.biases = biases

    def forward(self, dataIn):
        """
        Perform forward pass computation.
        
        Args:
            dataIn: Input data with shape (batch_size, sizeIn)
            
        Returns:
            Output activations with shape (batch_size, sizeOut)
        """
        self.setPrevIn(dataIn)
        output = np.dot(dataIn, self.weights) + self.biases
        self.setPrevOut(output)
        return output

    def gradient(self):
        """
        Compute the gradient for backpropagation.
        
        Returns:
            Gradient tensor for backpropagation
        """
        batch_size = self.getPrevIn().shape[0]
        return np.array([self.weights.T for _ in range(batch_size)])

    def backward(self, gradIn):
        """
        Perform backward pass to compute gradients.
        
        Args:
            gradIn: Gradient coming from the next layer
            
        Returns:
            Gradient to be passed to the previous layer
        """
        X = self.getPrevIn()
        self.gradW = np.dot(X.T, gradIn)
        self.gradB = np.sum(gradIn, axis=0, keepdims=True)
        return np.dot(gradIn, self.weights.T)

    def updateWeights(self, eta):
        """
        Update weights using computed gradients.
        
        Args:
            eta: Learning rate
        """
        if self.gradW is not None and self.gradB is not None:
            self.weights -= eta * self.gradW / self.gradW.shape[0]
            self.biases -= eta * self.gradB / self.gradB.shape[0]
            self.gradW = None
            self.gradB = None

    def updateWeightsWithMomentum(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, momentum=None, velocity=None):
        """
        Update weights using Adam-like optimization with momentum and adaptive learning rates.
        
        Args:
            learning_rate: Base learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            momentum: Dictionary of momentum values for weights and biases
            velocity: Dictionary of velocity values for weights and biases
        """
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
