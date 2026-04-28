from NeuralNetwork import Neural_Network as NN
from AutoGrad import Value as V
from Optimizer import Optimizer

inputs = [[V(0.90),V(0.80),V(0.10)],
          [V(0.25),V(0.20),V(0.85)],
          [V(0.70),V(0.55),V(0.35)],
          [V(0.45),V(0.75),V(0.20)],
          [V(0.15),V(0.10),V(0.95)],
          [V(0.80),V(0.35),V(0.45)]]
outputs = [[V(0.95),V(0.90),V(0.85)],
           [V(0.20),V(0.25),V(0.15)],
           [V(0.70),V(0.65),V(0.60)],
           [V(0.60),V(0.80),V(0.70)],
           [V(0.10),V(0.05),V(0.05)],
           [V(0.65),V(0.55),V(0.45)]]
# Inputs: sleep quality, exercise, screen-time load. Outputs: energy, focus, recovery.

model = NN([3,10,10,10,3])
optimizer = Optimizer(model)

for epoch in range(1000):
    loss = optimizer.loss(inputs,outputs)
    if epoch % 100 == 0:
        print(loss)
    optimizer.update_params(loss)

for sample in inputs:
    print(model.forward([sample]))

model.forward([[V(0.80),V(0.30),V(0.90)]]) # ivancho data
