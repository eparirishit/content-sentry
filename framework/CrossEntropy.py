import numpy as np

class CrossEntropy:
    def eval(self, Y, Yhat):
        eps = 1e-7
        return -np.mean(Y * np.log(Yhat + eps) + (1 - Y) * np.log(1 - Yhat + eps))
    
    def gradient(self, Y, Yhat):
        eps = 1e-7
        return (Yhat - Y) / (Yhat * (1 - Yhat) + eps)
