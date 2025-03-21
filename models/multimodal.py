import numpy as np
from framework import FullyConnectedLayer, LogisticSigmoidLayer, ReLULayer

class MultimodalModel:
    """
    Multimodal fusion model that combines image and text features.
    
    This model takes the outputs of the image and text networks,
    concatenates them, and performs additional processing to
    produce a final classification output.
    """
    def __init__(self, image_model, text_model, use_advanced_fusion=True):
        """
        Initialize the multimodal fusion model.
        
        Args:
            image_model: Model for processing image inputs
            text_model: Model for processing text inputs
            use_advanced_fusion: Whether to use multi-layer fusion (vs. simple)
        """
        self.image_model = image_model
        self.text_model = text_model
        
        # Add a more sophisticated fusion mechanism
        if use_advanced_fusion:
            self.fc1 = FullyConnectedLayer(128, 64)
            self.relu = ReLULayer()
            self.fc2 = FullyConnectedLayer(64, 1)
            self.sigmoid = LogisticSigmoidLayer()
            self.fc = self.fc1  # For backward compatibility
        else:
            self.fc = FullyConnectedLayer(128, 1)
            self.sigmoid = LogisticSigmoidLayer()
            
        self.training = True
        self.use_advanced_fusion = use_advanced_fusion

    def train(self):
        """Set model to training mode."""
        self.training = True
        self.image_model.training = True
        self.text_model.training = True

    def eval(self):
        """Set model to evaluation mode."""
        self.training = False
        self.image_model.training = False
        self.text_model.training = False

    def forward(self, image, text):
        """
        Perform forward pass through the full multimodal network.
        
        Args:
            image: Image input data
            text: Text input data
            
        Returns:
            Binary classification probability (0-1)
        """
        img_feat = self.image_model.forward(image)
        txt_feat = self.text_model.forward(text)
        combined = np.concatenate([img_feat, txt_feat], axis=1)
        
        # Advanced fusion with multiple layers
        if self.use_advanced_fusion:
            x = self.fc1.forward(combined)
            x = self.relu.forward(x)
            x = self.fc2.forward(x)
            return self.sigmoid.forward(x)
        else:
            # Original simple fusion
            return self.sigmoid.forward(self.fc.forward(combined))

    def backward(self, grad):
        """
        Perform backward pass through the full multimodal network.
        
        Args:
            grad: Gradient from loss function
        """
        grad = self.sigmoid.backward(grad)
        
        if self.use_advanced_fusion:
            grad = self.fc2.backward(grad)
            grad = self.relu.backward(grad)
            grad = self.fc1.backward(grad)
        else:
            grad = self.fc.backward(grad)
            
        grad_img, grad_txt = np.split(grad, 2, axis=1)
        self.image_model.backward(grad_img)
        self.text_model.backward(grad_txt)

    def update_weights(self, lr):
        """
        Update all weights in the full multimodal network.
        
        Args:
            lr: Learning rate
        """
        if self.use_advanced_fusion:
            self.fc1.updateWeights(lr)
            self.fc2.updateWeights(lr)
        else:
            self.fc.updateWeights(lr)
            
        self.image_model.update_weights(lr)
        self.text_model.update_weights(lr)