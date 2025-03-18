from framework.Layer import Layer

class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        self.setPrevIn(data)
        N = data.shape[0]
        flat = data.reshape(N, -1, order='F')
        self.setPrevOut(flat)
        return flat

    def gradient(self):
        return None

    def backward(self, gradIn):
        return gradIn.reshape(self.getPrevIn().shape, order='F')
