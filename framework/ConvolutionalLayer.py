import numpy as np
from framework.Layer import Layer

class ConvolutionalLayer(Layer):
    def __init__(self, num_input_channels=1, num_kernels=1, kernel_size=3, stride=1, padding=0):
        """
        Parameters:
          num_input_channels: number of channels in the input
          num_kernels: number of convolutional filters
          kernel_size: size of the (square) kernel
          stride: stride length
          padding: number of zeros to pad on each side
        """
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Initialize kernels: shape (num_kernels, num_input_channels, kernel_size, kernel_size)
        self.kernels = np.random.uniform(-1e-4, 1e-4, 
                                          size=(num_kernels, num_input_channels, kernel_size, kernel_size))
        # Initialize biases: shape (num_kernels,)
        self.biases = np.random.uniform(-1e-4, 1e-4, size=(num_kernels,))

    def forward(self, input_data):
        # Ensure input_data is 4D: (batch, channels, height, width)
        if input_data.ndim == 3:
            input_data = input_data[:, None, :, :]
        batch_size, channels, height, width = input_data.shape
        # Apply padding if needed
        if self.padding > 0:
            input_data = np.pad(input_data, 
                                ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant')
        # Compute output dimensions
        out_height = (input_data.shape[2] - self.kernel_size) // self.stride + 1
        out_width = (input_data.shape[3] - self.kernel_size) // self.stride + 1
        # Initialize output tensor: (batch_size, num_kernels, out_height, out_width)
        output = np.zeros((batch_size, self.num_kernels, out_height, out_width))
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = input_data[b, :, i*self.stride:i*self.stride+self.kernel_size, 
                                                   j*self.stride:j*self.stride+self.kernel_size]
                        output[b, k, i, j] = np.sum(region * self.kernels[k]) + self.biases[k]
        self.setPrevIn(input_data)
        self.setPrevOut(output)
        return output

    def gradient(self):
        # Full gradient computation is not implemented.
        return None

    def backward(self, gradIn):
        # Store the gradient for later use in updateKernels
        self.stored_gradient = gradIn
        return gradIn

    def updateKernels(self, gradIn, eta):
        """
        Updates the kernel weights using the gradient from the next layer.
        
        Parameters:
          gradIn: gradient from the next layer (shape: (batch, num_kernels, out_height, out_width))
                 If None, uses stored gradient from backward pass (if available)
          eta: learning rate
        """
        # Handle the case when gradIn is None
        if gradIn is None:
            # If we have a stored gradient from backward pass, use it
            if hasattr(self, 'stored_gradient') and self.stored_gradient is not None:
                gradIn = self.stored_gradient
            else:
                # If no gradient is available, we can't update
                return
        
        batch_size = self.getPrevIn().shape[0]
        k = self.kernel_size
        out_H = gradIn.shape[2]
        out_W = gradIn.shape[3]
        dK = np.zeros((self.num_kernels, self.num_input_channels, k, k))
        for n in range(batch_size):
            for kernel_index in range(self.num_kernels):
                for i in range(out_H):
                    for j in range(out_W):
                        patch = self.getPrevIn()[n, :, i*self.stride:i*self.stride+k, j*self.stride:j*self.stride+k]
                        dK[kernel_index] += patch * gradIn[n, kernel_index, i, j]
        # Update kernels and biases (averaged over batch)
        self.kernels -= eta * dK / batch_size
        dB = np.sum(gradIn, axis=(0,2,3)) / batch_size
        self.biases -= eta * dB
        
        # Clear stored gradient after update
        if hasattr(self, 'stored_gradient'):
            self.stored_gradient = None
    
    def update_weights(self, learning_rate):
        """
        Update weights method for API consistency with other layers.
        Calls updateKernels with the stored gradient.
        """
        self.updateKernels(None, learning_rate)
