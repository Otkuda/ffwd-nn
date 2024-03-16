import numpy as np

def nodeCost(outputActivation, expectedOutput):
    error = outputActivation - expectedOutput
    return error * error