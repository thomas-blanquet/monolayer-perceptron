import numpy as np


class Perceptron:
    def __init__(self, inputs):

        self.learningRate = 0.1
        self.weights = []
        self.threshold = None
        self.weightsWithThreshold = np.zeros(input + 1)
