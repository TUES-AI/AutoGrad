from AutoGrad import Value
import time

class GD_Optimizer:
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate # speed of learning, often between 0.01 and 0.0001

    def MSE_loss(self, linear_regression, inputs, labels): # Loss function
        loss = Value(0)
        for i in range(len(inputs)):
            prediction = linear_regression.forward(inputs[i])
            loss += (prediction - labels[i]) ** 2
        loss /= Value(len(inputs))
        print(f"Loss: {loss.value}")
        return loss

    def update_weights(self,linear_regression,loss): # gradient descent step
        loss.backward()
        linear_regression.w.value -= linear_regression.w.grad * self.learning_rate 
        linear_regression.b.value -= linear_regression.b.grad * self.learning_rate
        return linear_regression 

class Linear_refression:
    def __init__(self,w,b):
        self.w = Value(w)
        self.b = Value(b)
    def forward(self,x):
        y = (x * self.w + self.b)
        return y

if __name__ == "__main__":

    Eva = GD_Optimizer(learning_rate=0.05)

    BOBI = Linear_refression(0,-0)

    for i in range(100):

        inputs = [Value(1), Value(2),Value(3),Value(4)]
        labels = [Value(50),Value(55),Value(65),Value(70)]

        print("\nBackproping..\n")
        loss = Eva.MSE_loss(BOBI,inputs, labels)
        BOBI = Eva.update_weights(BOBI,loss)
        loss.zero_grads()
        print(f"Updated weights: w={BOBI.w.value}, b={BOBI.b.value}")
