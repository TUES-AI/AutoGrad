from AutoGrad import Value
import math
import random

class Neural_Network:
    def __init__(self,layer_sizes): # [2,3,1]
        self.layers = []
        for i in range(len(layer_sizes)-1):
            weights = []
            for j in range(layer_sizes[i]):
                row = []
                for k in range(layer_sizes[i+1]):
                    row.append(Value(random.uniform(-math.sqrt(6/layer_sizes[i]), math.sqrt(6/layer_sizes[i]))))
                weights.append(row)
            self.layers.append(weights)

    def mat_mul(self, A, B):
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(B[0])):
                sum = Value(0)
                for k in range(len(B)):
                    sum += A[i][k] * B[k][j]
                row.append(sum)
            result.append(row)
        return result

    def forward(self,x):
        for l in range(len(self.layers)):
            x = self.mat_mul(x,self.layers[l])
            if l < len(self.layers)-1:
                x = [[x[i][j].relu() for j in range(len(x[0]))] for i in range(len(x))]
        return x
