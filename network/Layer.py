import numpy as np
from Activation import sigmoid

class Layer:

    def __init__(self, n_nodes_in, n_nodes_out) -> None:
        # int, number of nodes in previous layer and number of nodes in this layer
        self.n_nodes_in = n_nodes_in
        self.n_nodes_out = n_nodes_out
        # array to store weight values for this layer
        self.weights = np.empty((n_nodes_in, n_nodes_out))
        self.costGradientW = np.empty((n_nodes_in, n_nodes_out))
        # array to store bias values for this layer 
        self.biases = np.empty(n_nodes_out)
        self.costGradientB = np.empty(n_nodes_out)

        self.initRandomWeights()

    def applyGradient(self, learnRate):
        for nodeOut in range(self.n_nodes_out):
            self.biases[nodeOut] -= self.costGradientB[nodeOut] * learnRate
            for nodeIn in range(self.n_nodes_in):
                self.weights[nodeIn, nodeOut] -= self.costGradientW[nodeIn, nodeOut]

    def calculateOutputs(self, inputs):
        activations = np.empty(self.n_nodes_out)

        for nodeOut in range(self.n_nodes_out):
            weightedInput = self.biases[nodeOut]   
            for nodeIn in range(self.n_nodes_in):
                weightedInput += inputs[nodeIn] * self.weights[nodeIn, nodeOut]
            activations[nodeOut] = sigmoid(weightedInput)
        
        return activations

    def initRandomWeights(self):
        for el in self.weights:
            el = np.random.normal()

    def nodeCost(self, outputActivation, expectedOutput):
        error = outputActivation - expectedOutput
        return error * error
    
    def costDerivative(self, outputActivation, expectedOutput):
        return 2 * (outputActivation - expectedOutput)
    

class NetworkLearnData:

    def __init__(self, layers) -> None:
        self.layerData = np.empty(len(layers))

        for i in range(len(layers)):
            self.layerData[i] = LayerLearnData(layers[i])


class LayerLearnData:

    def __init__(self, layer) -> None:
        self.weightedInputs = np.empty(layer.n_nodes_out)
        self.activations = np.empty(layer.n_nodes_out)
        self.nodeValues = np.empty(layer.n_nodes_out)