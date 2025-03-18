import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
  def __init__(self):
    self.__prevIn = []
    self.__prevOut = []
    self.training = True
  
  def setPrevIn(self, dataIn):
    self.__prevIn = dataIn
  
  def setPrevOut(self, out):
    self.__prevOut = out
  
  def getPrevIn(self):
    return self.__prevIn
  
  def getPrevOut(self):
    return self.__prevOut
  
  """
  def backward(self, gradIn):
    sg = self.gradient()
    gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))

    for n in range(gradIn.shape[0]):
        gradOut[n] = np.atleast_2d(gradIn[n])@sg[n]
    return gradOut
  """
  def backward(self, gradIn):
    sg = self.gradient()
    
    if(sg.ndim == 3):  #tensor coming back
        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))
        for i in range(gradIn.shape[0]):
            gradOut[i] = np.atleast_2d(gradIn[i])@np.atleast_2d(sg[i])
    else:
        gradOut = np.atleast_2d(gradIn)@sg
    
    return gradOut

  @abstractmethod
  def forward(self,dataIn):
    pass
 
  @abstractmethod
  def gradient(self):
    pass

