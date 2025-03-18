from framework import (
    EmbeddingLayer, TransposeLayer, Conv1DLayer,
    ReLULayer, GlobalMaxPooling1DLayer,
    FlatteningLayer, FullyConnectedLayer
)


class TextCNN:
    def __repr__(self):
        return "\n".join([
            "TextCNN Architecture:",
            *[f"- {layer.__class__.__name__}" for layer in self.layers]
        ])
    
    def __init__(self, vocab_size, max_length=20):
        self.layers = [
            EmbeddingLayer(vocab_size, 100),
            TransposeLayer((0, 2, 1)),
            Conv1DLayer(100, 128, 5),
            ReLULayer(),
            GlobalMaxPooling1DLayer(),
            FlatteningLayer(),
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