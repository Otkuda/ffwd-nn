import Layer
from utils.DataPoint import DataPoint

class NeuralNetwork:

    def __init__(self, layerSizes) -> None:
        self.layers = [Layer(layerSizes[i], layerSizes[i + 1]) for i in range(len(layerSizes) - 1)]

    def calculateOutputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculateOutputs(inputs)
        
        return inputs
    
    def classify(self, inputs):
        outputs = self.calculateOutputs(inputs)
        return outputs
    
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
    
    def updateAllGradients(self, dataPoint):
        self