import numpy as np
from simplenn import SimpleNeuralNetwork

network = SimpleNeuralNetwork()
print(network)
# print(network.weights)

train_inputs = np.array([[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0],[0,0,0],[0,0,1]])
train_outputs = np.array([[1,0,1,1,0,1,1,0]]).T
train_iterators = 80_000

network.train(train_inputs,train_outputs,train_iterators)
print(f"wytrenowane wagi:\n{network.weights}")
print("__________ predykcja ___________")
test_data = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])

for data in test_data:
    print(f"wynik dla {data} wynosi: {network.propagation(data)}")
