import numpy as np

class Neural_Network:
    def init_value(self):
        return np.random.uniform(-1, 1)

    def __init__(self, layers):
        self.layers = []

        for i in range(len(layers) - 1):
            input_size = layers[i]
            output_size = layers[i + 1]

            matrix = np.array([self.init_value() for _ in range(input_size * output_size)])
            matrix = matrix.reshape(input_size, output_size)
            self.layers.append(matrix)

        self.activations = []
        self.before_relu = []
        self.loss = 0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return x > 0

    def calculate_activations(self, x):
        self.activations = [x]
        self.before_relu = []

        for W in self.layers[:-1]:
            before_relu = x @ W
            self.before_relu.append(before_relu)

            x = self.relu(before_relu)
            self.activations.append(x)

        prediction = x @ self.layers[-1]
        self.activations.append(prediction)
        return prediction

    def forward(self, x):
        return self.calculate_activations(x)

    def calculate_loss(self, prediction, target):
        errors = prediction - target
        self.loss = np.sum(errors**2)
        return self.loss

    def calculate_loss_gradient(self, prediction, target):
        return 2 * (prediction - target)

    def backward(self, x, target):
        prediction = self.forward(x)
        self.calculate_loss(prediction, target)

        grad = self.calculate_loss_gradient(prediction, target)
        layer_grads = []

        for layer_index in range(len(self.layers) - 1, -1, -1):
            previous_activation = self.activations[layer_index]
            weights = self.layers[layer_index]

            grad_weights = previous_activation.reshape(-1, 1) @ grad.reshape(1, -1)
            layer_grads.append(grad_weights)

            grad = grad @ weights.T

            if layer_index > 0:
                grad = grad * self.relu_derivative(self.before_relu[layer_index - 1])

        layer_grads.reverse()
        return layer_grads

    def train(self, x, target, learning_rate=0.1):
        layer_grads = self.backward(x, target)

        for layer_index in range(len(self.layers)):
            self.layers[layer_index] -= learning_rate * layer_grads[layer_index]


if __name__ == "__main__":
    np.random.seed(0)

    nn = Neural_Network([2, 3, 1])

    x = np.array([1, 2])
    target = np.array([1])

    print(f"Before training: {nn.forward(x)}")
    print(f"Loss: {nn.calculate_loss(nn.forward(x), target)}")

    nn.train(x, target)

    print(f"After training: {nn.forward(x)}")
    print(f"Loss: {nn.calculate_loss(nn.forward(x), target)}")
