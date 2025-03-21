import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
  """
  Abstract base class for neural network layers.
  
  Defines the common interface that all layer implementations must follow,
  including forward pass, gradient computation, and backpropagation.
  """
  def __init__(self):
    """Initialize layer with empty storage for input and output values."""
    self.__prevIn = []
    self.__prevOut = []
    self.training = True
  
  def setPrevIn(self, dataIn):
    """Store the input data for later use in backpropagation."""
    self.__prevIn = dataIn
  
  def setPrevOut(self, out):
    """Store the output data for later use in backpropagation."""
    self.__prevOut = out
  
  def getPrevIn(self):
    """Retrieve the stored input data."""
    return self.__prevIn
  
  def getPrevOut(self):
    """Retrieve the stored output data."""
    return self.__prevOut
  
  def backward(self, gradIn):
    """
    Default implementation of backpropagation.
    
    Args:
        gradIn: Gradient flowing back from the next layer
        
    Returns:
        Gradient to be passed to the previous layer
    """
    sg = self.gradient()
    
    if(sg.ndim == 3):
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
    else:
        gradOut = np.atleast_2d(gradIn)@sg
    
    return gradOut

  @abstractmethod
  def forward(self,dataIn):
    """
    Perform forward pass computation for this layer.
    
    Args:
        dataIn: Input data to the layer
        
    Returns:
        Output of the layer's computation
    """
    pass
 
  @abstractmethod
  def gradient(self):
    """
    Compute the gradient for this layer's parameters.
    
    Returns:
        Gradient values for the layer's parameters
    """
    pass

