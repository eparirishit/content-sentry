import numpy as np
from framework.Layer import Layer

class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        self.grad_embeddings = np.zeros_like(self.embeddings)
        self.last_indices = None

    def forward(self, input_indices):
        self.setPrevIn(input_indices)
        self.last_indices = input_indices
        return self.embeddings[input_indices]

    def gradient(self):
        """Implement abstract method to return parameter gradients"""
        return self.grad_embeddings

    def backward(self, grad_output):
        """Handle gradient accumulation"""
        if self.last_indices is not None:
            # Reshape to 2D: (batch_size * sequence_length, embedding_dim)
            grad_flat = grad_output.reshape(-1, self.embeddings.shape[1])
            
            # Convert indices to 1D array
            indices_flat = self.last_indices.flatten()
            
            # Accumulate gradients using numpy's add.at
            np.add.at(self.grad_embeddings, indices_flat, grad_flat)
            
        return None  # No gradient for input indices

    def update_weights(self, learning_rate):
        self.embeddings -= learning_rate * self.grad_embeddings
        self.grad_embeddings.fill(0)