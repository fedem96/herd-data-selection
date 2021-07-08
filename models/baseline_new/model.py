""" per discriminare, viene applicata una soglia alla somma delle radici quadrate effettuate su ognuno degli 8000 numeri del tensore  """
import pickle

import numpy as np

class ModelBaseline:

    def __init__(self, file=None):
        self.threshold = 0.5
        self.max_sum = 0
        if file is not None:
            l = pickle.load(open(file, "rb"))
            self.threshold = l[0]
            self.max_sum = l[1]

    def fit(self, X_train, y_train, **kwargs):
        print(kwargs["t"])
        self.threshold = float(kwargs["t"])
        self.max_sum = np.max(np.sum((X_train/7 >= self.threshold).astype(int), axis=(1, 2, 3, 4)))

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            example = (X_test[i]/7 >= self.threshold).astype(int)
            if self.max_sum == 0:
                pred = np.sum(example)
            else:
                pred = np.sum(example) / self.max_sum
            if pred > 1:
                pred = 1
            predictions.append(pred)
        return predictions

    def save(self, file):
        l = [self.threshold, self.max_sum]
        pickle.dump(l, open(file, "wb"))


def get_model(file=None):
    return ModelBaseline(file)
