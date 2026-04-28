from NeuralNetwork import Neural_Network as NN
from AutoGrad import Value as V

class Optimizer:
    def __init__(self,model:NN,lr:float=0.01):
        self.model=model
        self.lr=lr
    
    def loss(self,xs,ys):
        loss = V(0)
        for x,y in zip(xs,ys):
            pred = self.model.forward([x])
            targets = y if isinstance(y,list) else [y]
            for i,target in enumerate(targets):
                loss += (pred[0][i]+(target*V(-1)))**2
        return loss/V(len(xs))

    def update_params(self,loss):
        loss.zero_grads()
        loss.backward()
        for layer in self.model.layers:
            for row in layer:
                for param in row:
                    param.value -= self.lr * param.grad
