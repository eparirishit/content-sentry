from framework import (
    EmbeddingLayer, TransposeLayer, Conv1DLayer,
    ReLULayer, GlobalMaxPooling1DLayer,
    FlatteningLayer, FullyConnectedLayer, BatchNormalizationLayer
)

class TextCNN:
    """
    CNN model for processing text data in multimodal classification.
    
    Implements a convolutional neural network for text feature extraction,
    using word embeddings and n-gram capturing filters.
    """
    def __repr__(self):
        """Return string representation of model architecture."""
        return "\n".join([
            "TextCNN Architecture:",
            *[f"- {layer.__class__.__name__}" for layer in self.layers]
        ])
    
    def __init__(self, vocab_size, max_length=20, embedding_dim=128, use_batchnorm=True):
        """
        Initialize the text CNN model.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimensionality of word embeddings
            use_batchnorm: Whether to use batch normalization
        """
        self.device = None
        
        self.layers = [
            EmbeddingLayer(vocab_size, embedding_dim),
            TransposeLayer((0, 2, 1)),
        ]
        
        # Multiple parallel convolutional filters with different kernel sizes
        filter_sizes = [3, 4, 5]  # capture 3-grams, 4-grams, and 5-grams
        num_filters = 64  # filters per size
        
        # First conv path (3-gram)
        conv1 = [
            Conv1DLayer(embedding_dim, num_filters, filter_sizes[0]),
            ReLULayer(),
            GlobalMaxPooling1DLayer(),
        ]
        
        # Second conv path (4-gram)
        conv2 = [
            Conv1DLayer(embedding_dim, num_filters, filter_sizes[1]),
            ReLULayer(),
            GlobalMaxPooling1DLayer(),
        ]
        
        # Third conv path (5-gram)
        conv3 = [
            Conv1DLayer(embedding_dim, num_filters, filter_sizes[2]),
            ReLULayer(),
            GlobalMaxPooling1DLayer(),
        ]
        
        # Add all convolutional paths
        self.layers.extend(conv1)
        
        # Add fully connected layers
        self.layers.extend([
            FlatteningLayer(),
            FullyConnectedLayer(num_filters, 64),  
        ])
        
        if use_batchnorm:
            self.layers.append(BatchNormalizationLayer(64))
            
        self.layers.append(ReLULayer())

    def forward(self, x):
        """
        Perform forward pass through the network.
        
        Args:
            x: Input text indices, shape (batch_size, sequence_length)
            
        Returns:
            Text feature representation, shape (batch_size, 64)
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