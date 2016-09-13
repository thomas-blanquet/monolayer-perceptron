import numpy as np
from math import sqrt


class Perceptron:
    def __init__(self, input):
        self.input = input  # Vector's dimension
        self.learningRate = 0.1  # Learning rate
        self.weightsForTrain = np.zeros(input + 1)  # + 1 for threshold calculation

        # defined after train
        self.threshold = None
        self.weights = None

    def train(self, X, Y):
        X = np.insert(X, X.shape[1], -1, axis=1)  # insert -1 at the end to calculate the threshold
        while not self.is_train(X, Y):
            for x, y in zip(X, Y):
                predicted = self.predict_for_train([x])[0]  # predict for each element one by one
                if predicted == y:
                    continue
                for i in range(len(x)):
                    self.weightsForTrain[i] += self.learningRate * (y - predicted) * x[i]  # correction of weights
        self.threshold = self.weightsForTrain[-1]
        self.weights = self.weightsForTrain[:-1]

    def is_train(self, X, Y):
        """
        Calculate if predictions of Perceptron are equals to expected output, using vector
        subtraction and norm calculation
        """
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


# Number representation
train_X = np.array([
    [1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1],
])

# Expected output
train_Y = np.array([
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
])

# Parameter of Perceptron is dimension on vectors
p = Perceptron(7)
p.train(train_X, train_Y)
print(p.predict([[1, 1, 1, 0, 0, 0, 0]]))
