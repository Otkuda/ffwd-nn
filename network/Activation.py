import numpy as np

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def sigmoidDerivative(input):
    activation = sigmoid(input)
    return activation * (1 - activation)