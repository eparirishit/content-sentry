from framework import (
    ConvolutionalLayer, MaxPoolingLayer, ReLULayer,
    FlatteningLayer, FullyConnectedLayer, BatchNormalizationLayer
)

class ImageCNN:
    def __repr__(self):
        return "\n".join([
            "ImageCNN Architecture:",
            *[f"- {layer.__class__.__name__}" for layer in self.layers]
        ])
    
    def __init__(self, config):
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
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_weights(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(lr)
            elif hasattr(layer, 'updateKernels'):
                layer.updateKernels(None, lr)  # For ConvolutionalLayer

    def update_weights_with_momentum(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, momentum=None, velocity=None):
        """Update all layer weights using Adam-like optimization"""
        for layer in self.layers:
            if hasattr(layer, 'updateWeightsWithMomentum'):
                layer.updateWeightsWithMomentum(learning_rate, beta1, beta2, epsilon, momentum, velocity)
            elif hasattr(layer, 'update_weights'):  # Fallback for layers without momentum support
                layer.update_weights(learning_rate)