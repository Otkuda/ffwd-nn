import numpy as np
import Layer
from utils.DataPoint import DataPoint

class NeuralNetwork:

    def __init__(self, layerSizes) -> None:
        self.layers = [Layer(layerSizes[i], layerSizes[i + 1]) for i in range(len(layerSizes) - 1)]
        self.batchLearnData = None

    def calculateOutputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculateOutputs(inputs)
        
        return inputs
    
    def classify(self, inputs):
        outputs = self.calculateOutputs(inputs)
        return outputs
    

    def learn(self, trainingData, learnRate):
        if (self.batchLearnData == None or len(self.batchLearnData) != len(trainingData)):
            self.batchLearnData = np.empty(len(trainingData))
            for i in range(len(self.batchLearnData)):
                self.batchLearnData[i] = NetworkLearnData(self.layers)

        for i in range(len(trainingData)):
            self.updateAllGradients(trainingData[i], self.batchLearnData[i])

        for i in range(len(self.layers)):
            self.layers[i].applyGradient(learnRate / len(trainingData))
        

    def costPoint(self, dataPoint):
        outputs = self.calculateOutputs(dataPoint.inputs)
        outputLayer = self.layers[len(self.layers) - 1]
        cost = 0
        for nodeOut in range(len(outputs)):
            cost += outputLayer.nodeCost(outputs[nodeOut], dataPoint.expectedOutputs[nodeOut])
        
        return cost
    
    def cost(self, data):
        totalCost = 0

        for dataPoint in data:
            totalCost += self.costPoint(dataPoint)
        return totalCost / len(data)
    
    def updateAllGradients(self, dataPoint, learnData):
        inputToNextLayer = dataPoint.inputs

        for i in range(len(self.layers)):
            inputToNextLayer = self.layers[i].calculateOutputs(inputToNextLayer, learnData.layerData[i])

        # backpropagation start
        lastLayerIndex = len(self.layers) - 1
        lastLayer = self.layers[lastLayerIndex]
        lastLearnData = learnData.layerData[lastLayerIndex]

        lastLayer.calculateOutputLayerNodeValues(lastLearnData, dataPoint.expectedOutputs)
        lastLayer.updateGradients(lastLearnData)

        i = lastLayerIndex - 1
        while(i > 0):
            layerLearnData = learnData.layerData[i]
            hiddenLayer = self.layers[i]

            hiddenLayer.calculateHiddenLayerNodeValues(layerLearnData, self.layers[i+1], learnData.layerData[i+1].nodeValues)
            hiddenLayer.updateGradients(layerLearnData)


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