import numpy as np

class DataPoint:

    def __init__(self, inputs, label, nLabels=26) -> None:
        self.inputs = inputs
        self.label = label
        self.expectedOutputs = self.createOneHot(label, nLabels)

    def createOneHot(self, index, nLabels):
        oneHot = np.zeros(nLabels)
        oneHot[index] = 1
        return oneHot