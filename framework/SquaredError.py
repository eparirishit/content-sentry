import numpy as np

class SquaredError:
    def eval(self, Y, Yhat):
        return np.mean((Y - Yhat) ** 2)

    def gradient(self, Y, Yhat):
        return -2 * (Y - Yhat)
