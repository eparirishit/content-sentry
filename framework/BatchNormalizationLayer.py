import numpy as np
from framework.Layer import Layer

class BatchNormalizationLayer(Layer):
    """
    Batch Normalization layer implementation.
    
    Normalizes inputs to have zero mean and unit variance,
    then applies learnable scale and shift parameters.
    """
    def __init__(self, input_dim, momentum=0.99, epsilon=1e-8):
        """
        Initialize a batch normalization layer.
        
        Args:
            input_dim: Dimensionality of input features
            momentum: Momentum for running mean and variance updates
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Running statistics for inference
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        
        # Parameters to learn
        self.gamma = np.ones(input_dim)
        self.beta = np.zeros(input_dim)
        
        # Gradients
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x):
        """
        Perform batch normalization forward pass.
        
        Args:
            x: Input data with shape (batch_size, input_dim)
            
        Returns:
            Normalized and scaled output with same shape as input
        """
        self.setPrevIn(x)
        
        if self.training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0) + self.epsilon
            
            # Update running statistics for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var)
            
            # Cache for backward
            self.cache = (x_norm, batch_mean, batch_var, x)
        else:
            # Use running statistics in inference mode
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var)
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        self.setPrevOut(out)
        return out

    def gradient(self):
        """
        Implementation of abstract method.
        Not used directly - backward handles all gradient computation.
        """
        return None

    def backward(self, grad_output):
        """
        Backward pass for batch normalization.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient to pass to previous layer
        """
        # Get cached values
        x_norm, batch_mean, batch_var, x = self.cache
        N = x.shape[0]
        
        # Gradients with respect to gamma and beta
        self.dgamma = np.sum(grad_output * x_norm, axis=0)
        self.dbeta = np.sum(grad_output, axis=0)
        
        # Gradient with respect to normalized input
        dx_norm = grad_output * self.gamma
        
        # Gradient with respect to input x
        std_inv = 1. / np.sqrt(batch_var)
        dx = (1. / N) * std_inv * (
            N * dx_norm 
            - np.sum(dx_norm, axis=0)
            - x_norm * np.sum(dx_norm * x_norm, axis=0)
        )
        
        return dx
    
    def update_weights(self, learning_rate):
        """
        Update gamma and beta parameters using computed gradients.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta
        
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)