import numpy as np
from framework import FullyConnectedLayer, LogisticSigmoidLayer

class MultimodalModel:
    def __init__(self, image_model, text_model):
        self.image_model = image_model
        self.text_model = text_model
        self.fc = FullyConnectedLayer(128, 1)
        self.sigmoid = LogisticSigmoidLayer()
        self.training = True  # Track training mode

    def train(self):
        """Set model in training mode"""
        self.training = True
        self.image_model.training = True
        self.text_model.training = True

    def eval(self):
        """Set model in evaluation mode"""
        self.training = False
        self.image_model.training = False
        self.text_model.training = False

    def forward(self, image, text):
        img_feat = self.image_model.forward(image)
        txt_feat = self.text_model.forward(text)
        combined = np.concatenate([img_feat, txt_feat], axis=1)
        return self.sigmoid.forward(self.fc.forward(combined))

    def backward(self, grad):
        grad = self.sigmoid.backward(grad)
        grad = self.fc.backward(grad)
        grad_img, grad_txt = np.split(grad, 2, axis=1)
        self.image_model.backward(grad_img)
        self.text_model.backward(grad_txt)

    def update_weights(self, lr):
        self.fc.update_weights(lr)
        self.image_model.update_weights(lr)
        self.text_model.update_weights(lr)