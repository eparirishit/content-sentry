from framework import (
    ConvolutionalLayer, MaxPoolingLayer, ReLULayer,
    FlatteningLayer, FullyConnectedLayer, BatchNormalizationLayer
)

class ImageCNN:
    """
    CNN model for processing image data in multimodal classification.
    
    Creates a configurable convolutional neural network for image feature extraction,
    with options for different depths, channel configurations, and normalization.
    """
    def __repr__(self):
        """Return string representation of model architecture."""
        return "\n".join([
            "ImageCNN Architecture:",
            *[f"- {layer.__class__.__name__}" for layer in self.layers]
        ])
    
    def __init__(self, config):
        """
        Initialize the image CNN model.
        
        Args:
            config: Dictionary with configuration parameters:
                - grayscale: Whether to use grayscale (1 channel) or RGB (3 channels)
                - img_size: Input image size (square)
                - num_conv_blocks: Number of convolutional blocks to use
                - num_kernels: Base number of kernels per convolutional layer
                - use_batchnorm: Whether to use batch normalization
                - increase_channels: Whether to double channels in deeper blocks
        """
        self.layers = []
        self.device = None
        in_channels = 1 if config['grayscale'] else 3
        img_size = config['img_size']
        use_batchnorm = config.get('use_batchnorm', True)  # Default to True
        
        # Convolutional blocks
        for i in range(config['num_conv_blocks']):
            out_channels = config['num_kernels'] * (2**i if config.get('increase_channels', False) else 1)
            self.layers.append(ConvolutionalLayer(in_channels, out_channels, 3, padding=1))
            
            # Add batch normalization if enabled
            if use_batchnorm:
                # For 4D tensors (batch, channels, height, width)
                self.layers.append(BatchNormalizationLayer(out_channels))
                
            self.layers.append(ReLULayer())
            self.layers.append(MaxPoolingLayer(2, 2))
            in_channels = out_channels
            img_size //= 2

        # Calculate flattened feature size
        flattened_size = in_channels * img_size * img_size
        
        # Fully connected layers with dropout (first layer)
        self.layers.append(FlatteningLayer())
        self.layers.append(FullyConnectedLayer(flattened_size, 128))
        if use_batchnorm:
            self.layers.append(BatchNormalizationLayer(128))
        self.layers.append(ReLULayer())
        
        # Output layer
        self.layers.append(FullyConnectedLayer(128, 64))

    def forward(self, x):
        """
        Perform forward pass through the network.
        
        Args:
            x: Input image data, shape (batch_size, channels, height, width)
            
        Returns:
            Image feature representation, shape (batch_size, 64)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """
        Perform backward pass to compute gradients.
        
        Args:
            grad: Gradient from next layer/component
            
        Returns:
            Gradient with respect to input
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_weights(self, lr):
        """
        Update all layer weights using computed gradients.
        
        Args:
            lr: Learning rate
        """
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(lr)
            elif hasattr(layer, 'updateKernels'):
                layer.updateKernels(None, lr)

    def update_weights_with_momentum(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, momentum=None, velocity=None):
        """
        Update all layer weights using Adam-like optimization.
        
        Args:
            learning_rate: Base learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            momentum: Dictionary of momentum values
            velocity: Dictionary of velocity values
        """
        for layer in self.layers:
            if hasattr(layer, 'updateWeightsWithMomentum'):
                layer.updateWeightsWithMomentum(learning_rate, beta1, beta2, epsilon, momentum, velocity)
            elif hasattr(layer, 'update_weights'):
                layer.update_weights(learning_rate)