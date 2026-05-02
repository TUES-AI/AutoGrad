import numpy as np

class Neural_Network:
    def init_value(self):
        return np.random.uniform(-1,1)

    def __init__(self,layers): # [2,3,1]
        self.layers = []
        for i in range(len(layers)-1):
            matrix = np.array([self.init_value() for _ in range(layers[i]*layers[i+1])])
            # [X, Y, Z,...]
            matrix = matrix.reshape(layers[i],layers[i+1])
            # [[X,Y],[Z,W],...]
            self.layers.append(matrix)

    def forward(self, x):
        for W in self.layers[:-1]:
            x = np.maximum(0, x @ W) # ReLU(x @ W)
        y = x @ self.layers[-1] # x @ W
        return y

# nn = Neural_Network([784,16,16,10])
#
# import time
# start = time.perf_counter()
#
# for i in range(1_000_000):
#     nn.forward(np.random.rand(784))
#
# end = time.perf_counter()
# total = end - start
# print(f"Total: {total:.6f} s")
