import keras
from keras.layers import Dense

# Part 1: First hidden layer (5 inputs, 8 neurons, sigmoid)
layer1 = Dense(8, input_dim=5, activation='sigmoid')
print("Layer 1 config:")
print(layer1.get_config())
print("\n")

# Part 2: Hidden layer 2 (4 neurons, sigmoid)
layer2 = Dense(4, activation='sigmoid')
print("Layer 2 config:")
print(layer2.get_config())
print("\n")

# Part 3: Output layer (1 neuron, sigmoid)
layer3 = Dense(1, activation='sigmoid')
print("Layer 3 config:")
print(layer3.get_config())
