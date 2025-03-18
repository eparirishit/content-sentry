import numpy as np
from framework.Layer import Layer

class MaxPoolingLayer(Layer):
    def __init__(self, pool_size, stride):
        """
        Initialize the max pooling layer.
        Parameters:
          pool_size: int, the height and width of the pooling window.
          stride: int, the stride of the pooling window.
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, data):
        self.setPrevIn(data)
        pool_size = self.pool_size
        stride = self.stride

        if data.ndim == 3:
            # Data shape: (N, H, W)
            N, H, W = data.shape
            out_H = (H - pool_size) // stride + 1
            out_W = (W - pool_size) // stride + 1
            output = np.zeros((N, out_H, out_W))
            # Store max indices for each sample
            self.max_indices = np.zeros((N, out_H, out_W, 2), dtype=int)
            
            for n in range(N):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * stride
                        h_end = h_start + pool_size
                        w_start = j * stride
                        w_end = w_start + pool_size
                        window = data[n, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        output[n, i, j] = max_val
                        index = np.argmax(window)
                        index_2d = (index // pool_size, index % pool_size)
                        self.max_indices[n, i, j, :] = (h_start + index_2d[0], w_start + index_2d[1])
        elif data.ndim == 4:
            # Data shape: (N, C, H, W)
            N, C, H, W = data.shape
            out_H = (H - pool_size) // stride + 1
            out_W = (W - pool_size) // stride + 1
            output = np.zeros((N, C, out_H, out_W))
            # Store max indices per channel
            self.max_indices = np.zeros((N, C, out_H, out_W, 2), dtype=int)
            
            for n in range(N):
                for c in range(C):
                    for i in range(out_H):
                        for j in range(out_W):
                            h_start = i * stride
                            h_end = h_start + pool_size
                            w_start = j * stride
                            w_end = w_start + pool_size
                            window = data[n, c, h_start:h_end, w_start:w_end]
                            max_val = np.max(window)
                            output[n, c, i, j] = max_val
                            index = np.argmax(window)
                            index_2d = (index // pool_size, index % pool_size)
                            self.max_indices[n, c, i, j, :] = (h_start + index_2d[0], w_start + index_2d[1])
        else:
            raise ValueError("Input data must be 3D or 4D")
            
        self.setPrevOut(output)
        return output

    def gradient(self):
        # Not used in this implementation; backward directly uses stored indices.
        return None

    def backward(self, gradIn):
        # Backward propagation for max pooling.
        # Supports both 3D and 4D input based on how forward() was computed.
        input_data = self.getPrevIn()
        if input_data.ndim == 3:
            N, H, W = input_data.shape
            dX = np.zeros_like(input_data)
            N_out, H_out, W_out = gradIn.shape
            for n in range(N_out):
                for i in range(H_out):
                    for j in range(W_out):
                        idx = self.max_indices[n, i, j]
                        dX[n, idx[0], idx[1]] += gradIn[n, i, j]
        elif input_data.ndim == 4:
            N, C, H, W = input_data.shape
            dX = np.zeros_like(input_data)
            N_out, C_out, H_out, W_out = gradIn.shape
            for n in range(N_out):
                for c in range(C_out):
                    for i in range(H_out):
                        for j in range(W_out):
                            idx = self.max_indices[n, c, i, j]
                            dX[n, c, idx[0], idx[1]] += gradIn[n, c, i, j]
        else:
            raise ValueError("Input data must be 3D or 4D")
        return dX
