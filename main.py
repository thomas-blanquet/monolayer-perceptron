import numpy as np
from math import sqrt


class Perceptron:
    def __init__(self, input):
        self.input = input
        self.learningRate = 0.1
        self.weights = None
        self.threshold = None
        self.weightsForTrain = np.zeros(input + 1)

    def train(self, X, Y):
        X = np.insert(X, X.shape[1], -1, axis=1)
        while not self.is_train(X, Y):
            for x, y in zip(X, Y):
                predicted = self.predict_for_train([x])[0]
                if predicted == y:
                    continue
                for i in range(len(X) - 1):
                    self.weightsForTrain[i] += self.learningRate * (y - predicted) * x[i]
        self.threshold = self.weightsForTrain[-1]
        self.weights = self.weightsForTrain[:-1]

    def is_train(self, X, Y):
        predicted = self.predict_for_train(X)
        result = predicted - Y
        norm = 0
        for x in result:
            norm += x ** 2
        return False if sqrt(norm) != 0 else True

    def predict_for_train(self, X):
        return [1 if np.dot(x, self.weightsForTrain) >= 0 else 0 for x in X]

    def predict(self, X):
        return [1 if np.dot(x, self.weights) >= self.threshold else 0 for x in X]

train_X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])
train_Y = np.array([
    0,
    0,
    0,
    1,
])

p = Perceptron(2)
p.train(train_X, train_Y)
print(p.predict(train_X))