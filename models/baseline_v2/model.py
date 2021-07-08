""" per discriminare, viene applicata una soglia alla somma delle radici quadrate effettuate su ognuno degli 8000 numeri del tensore  """
import pickle

import numpy as np

class ModelBaseline:

    def __init__(self, file=None):
        self.threshold = 0.002
        self.max_sum = 0
        self.max_val = 0
        if file is not None:
            self.max_sum = pickle.load(open(file, "rb"))

    def fit(self, X_train, y_train, **args):
        self.max_val = np.max(X_train)
        self.max_sum = np.max(np.sum((X_train >= self.threshold).astype(int), axis=(1, 2, 3, 4)))

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            print(np.sum(X_test[i]))
            example = (X_test[i]/self.max_val >= self.threshold).astype(int)
            print(np.sum(example))
            print()
            if self.max_sum == 0:
                pred = np.sum(example)
            else:
                pred = np.sum(example) / self.max_sum
            if pred > 1:
                pred = 1
            predictions.append(pred)
        return predictions

    def save(self, file):
        pickle.dump(self.max_sum, open(file, "wb"))


def get_model(file=None):
    return ModelBaseline(file)
