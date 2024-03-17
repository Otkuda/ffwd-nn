import numpy as np
from Activation import sigmoid, sigmoidDerivative

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
    
    # passing data through NN and saving data to learnData obj
    def calculateOutputs(self, inputs, learnData):
        learnData.inputs = inputs

        for nodeOut in range(self.n_nodes_out):
            weightedInput = self.biases[nodeOut] 
            for nodeIn in range(self.n_nodes_in):
                weightedInput += inputs[nodeIn] * self.weights[nodeIn, nodeOut]
            learnData.weightedInputs[nodeOut] = weightedInput
            learnData.activations[nodeOut] = sigmoid(weightedInput)

        return learnData.activations

    def calculateOutputLayerNodeValues(self, layerLearnData, expectedOutputs):
        for i in range(len(len(layerLearnData.nodeValues))):
            costDerivative = self.costDerivative(layerLearnData.activations[i], expectedOutputs[i])
            activationDerivative = sigmoidDerivative(layerLearnData.weightedInputs[i])
            layerLearnData.nodeValues = costDerivative * activationDerivative

    def calculateHiddenLayerNodeValues(self, layerLearnData, oldLayer, oldNodeValues):
        for newNodeIndex in range(self.n_nodes_out):
            newNodeValue = 0
            for oldNodeIndex in range(len(oldNodeValues)):
                weightedInputDerivative = oldLayer.weights[newNodeIndex, oldNodeIndex]
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex]
            newNodeValue *= sigmoidDerivative(layerLearnData.weightedInputs[newNodeIndex])
            layerLearnData.nodeValues[newNodeIndex] = newNodeValue

    def updateGradients(self, layerLearnData):
        for nodeOut in range(self.n_nodes_out):
            nodeValue = layerLearnData.nodeValues[nodeOut]
            for nodeIn in range(self.n_nodes_in):
                self.costGradientW[nodeIn, nodeOut] += layerLearnData.inputs[nodeIn] * nodeValue
            self.costGradientB += nodeValue

    def initRandomWeights(self):
        for el in self.weights:
            el = np.random.normal()

    def nodeCost(self, outputActivation, expectedOutput):
        error = outputActivation - expectedOutput
        return error * error
    
    def costDerivative(self, outputActivation, expectedOutput):
        return 2 * (outputActivation - expectedOutput)
    