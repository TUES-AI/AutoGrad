import numpy as np

from NeuralNetwork_grads import Neural_Network


class Optimizer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def loss(self, xs, ys):
        total_loss = 0

        for x, y in zip(xs, ys):
            prediction = self.model.forward(x)
            total_loss += self.model.calculate_loss(prediction, y)

        return total_loss / len(xs)

    def calculate_grads(self, xs, ys):
        total_grads = [np.zeros_like(layer) for layer in self.model.layers]

        for x, y in zip(xs, ys):
            sample_grads = self.model.backward(x, y)

            for layer_index in range(len(total_grads)):
                total_grads[layer_index] += sample_grads[layer_index]

        for layer_index in range(len(total_grads)):
            total_grads[layer_index] /= len(xs)

        return total_grads

    def update_params(self, layer_grads):
        for layer_index in range(len(self.model.layers)):
            self.model.layers[layer_index] -= self.learning_rate * layer_grads[layer_index]

    def step(self, xs, ys):
        starting_loss = self.loss(xs, ys)
        layer_grads = self.calculate_grads(xs, ys)
        self.update_params(layer_grads)
        return starting_loss


if __name__ == "__main__":
    np.random.seed(1)

    xs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 1],
        [1, 2],
    ])
    ys = np.array([
        [0],
        [1],
        [1],
        [2],
        [3],
        [3],
    ])

    network = Neural_Network([2, 4, 1])
    optimizer = Optimizer(network, learning_rate=0.01)

    print(f"Starting loss: {optimizer.loss(xs, ys)}")

    for epoch in range(200):
        optimizer.step(xs, ys)

    print(f"Final loss: {optimizer.loss(xs, ys)}")

    for x, y in zip(xs, ys):
        prediction = network.forward(x)
        print(f"Input: {x}, prediction: {prediction}, target: {y}")
