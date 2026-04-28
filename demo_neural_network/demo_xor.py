from NeuralNetwork import Neural_Network as NN
from AutoGrad import Value as V

inputs = [[V(0),V(0)],
          [V(0),V(1)],
          [V(1),V(0)],
          [V(1),V(1)]]
outputs = [V(0),
           V(1),
           V(1),
           V(0)]
# XOR gate

model = NN([2,10,10,10,1])
optimizer = Optimizer(model)

for epoch in range(100):
    loss = optimizer.loss(inputs,outputs)
    print(loss)
    optimizer.update_params(loss)

print(model.forward([inputs[0]]))
print(model.forward([inputs[1]]))
print(model.forward([inputs[2]]))
print(model.forward([inputs[3]]))
