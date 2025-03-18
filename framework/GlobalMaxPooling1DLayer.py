import numpy as np
from framework.Layer import Layer

class GlobalMaxPooling1DLayer(Layer):
    def __init__(self):
        super().__init__()
        self.max_indices = None

    def forward(self, x):
        """Input shape: (batch, channels, sequence)"""
        self.setPrevIn(x)
        self.max_indices = np.argmax(x, axis=2)
        output = np.take_along_axis(x, self.max_indices[:, :, None], axis=2).squeeze()
        self.setPrevOut(output)
        return output

    def gradient(self):
        """Required by abstract base class (no parameters to update)"""
        return None

    def backward(self, grad_output):
        """Gradient propagation for max positions"""
        input_shape = self.getPrevIn().shape
        grad_input = np.zeros(input_shape)
        
        for b in range(grad_output.shape[0]):
            for c in range(grad_output.shape[1]):
                grad_input[b, c, self.max_indices[b, c]] = grad_output[b, c]
        
        return grad_input