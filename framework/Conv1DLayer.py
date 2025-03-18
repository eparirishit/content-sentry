import numpy as np
from framework.Layer import Layer

class Conv1DLayer(Layer):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # He initialization for ReLU networks
        self.weights = np.random.randn(out_channels, in_channels, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size))
        self.bias = np.zeros(out_channels)
        self.last_input = None
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        
    def forward(self, x):
        self.setPrevIn(x)
        batch_size, in_ch, seq_len = x.shape
        out_seq = seq_len - self.weights.shape[2] + 1
        self.last_input = x
        output = np.zeros((batch_size, self.weights.shape[0], out_seq))
        
        # Convolution operation
        for b in range(batch_size):
            for oc in range(self.weights.shape[0]):
                for i in range(out_seq):
                    window = x[b, :, i:i+self.weights.shape[2]]
                    output[b, oc, i] = np.sum(window * self.weights[oc]) + self.bias[oc]
        
        self.setPrevOut(output)
        return output

    def gradient(self):
        """Implement abstract method"""
        return self.grad_weights, self.grad_bias

    def backward(self, grad_output):
        batch_size, _, out_seq = grad_output.shape
        grad_input = np.zeros_like(self.last_input)
        
        # Compute gradients
        for b in range(batch_size):
            for oc in range(self.weights.shape[0]):
                for i in range(out_seq):
                    window = self.last_input[b, :, i:i+self.weights.shape[2]]
                    self.grad_weights[oc] += grad_output[b, oc, i] * window
                    self.grad_bias[oc] += grad_output[b, oc, i]
                    grad_input[b, :, i:i+self.weights.shape[2]] += self.weights[oc] * grad_output[b, oc, i]
        
        return grad_input

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias
        self.grad_weights.fill(0)
        self.grad_bias.fill(0)