from framework import (
    ConvolutionalLayer, MaxPoolingLayer, ReLULayer,
    FlatteningLayer, FullyConnectedLayer
)

class ImageCNN:
    def __repr__(self):
        return "\n".join([
            "ImageCNN Architecture:",
            *[f"- {layer.__class__.__name__}" for layer in self.layers]
        ])
    
    def __init__(self, config):
        self.layers = []
        in_channels = 1 if config['grayscale'] else 3
        img_size = config['img_size']
        
        # Convolutional blocks
        for _ in range(config['num_conv_blocks']):
            self.layers += [
                ConvolutionalLayer(in_channels, config['num_kernels'], 3, padding=1),
                ReLULayer(),
                MaxPoolingLayer(2, 2)
            ]
            in_channels = config['num_kernels']
            img_size //= 2

        # Fully connected
        self.layers += [
            FlatteningLayer(),
            FullyConnectedLayer(in_channels * img_size**2, 128),
            ReLULayer(),
            FullyConnectedLayer(128, 64)
        ]

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