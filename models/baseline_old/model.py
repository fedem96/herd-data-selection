""" pre discriminare, viene applicata una soglia alla somma degli 8000 numeri del tensore """
import pickle

import numpy as np


class ModelBaseline:

    def __init__(self, file=None):
        self.threshold = 0
        self.max_sum = 0
        if file is not None:
            self.max_sum = pickle.load(open(file, "rb"))

    def fit(self, X_train, y_train, **args):
        self.max_sum = np.max(np.sum(X_train, axis=(1, 2, 3, 4)))

    def predict(self, X_test):
        predictions = []
        for example in X_test:
            pred = np.sum(example) / self.max_sum
            if pred > 1:
                pred = 1
            predictions.append(pred)
        return predictions

    def save(self, file):
        pickle.dump(self.max_sum, open(file, "wb"))

    def evaluate(self, X, y):

        if len(X) != len(y):
            print("Error")
            return None

        tp = 0  # true positives
        fp = 0  # false positives
        fn = 0  # false negatives
        for i in range(len(y)):
            example = X[i]
            label = y[i]
            if np.sum(example) >= self.threshold:
                if label:
                    tp += 1
                else:
                    fp += 1
            else:
                if label:
                    fn += 1

        if tp+fp > 0:
            precision = tp/(tp+fp)
        else:
            precision = 0

        if tp+fn > 0:
            recall = tp/(tp+fn)
        else:
            recall = 0

        return precision, recall

    def get_thresholds(self, X):
        sums = np.sum(X, axis=(1, 2, 3, 4))
        min = int(np.min(sums))
        max = int(np.max(sums) + 1)
        step = (max - min) / 20
        return np.arange(min, max, step)


def get_model(file=None):
    return ModelBaseline(file)
