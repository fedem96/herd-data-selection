""" modello provato, senza risultati interessanti """

import numpy as np


class ModelThresholdBinary:

    @staticmethod
    def bounds():
        return 0

    @staticmethod
    def evaluate(X, y, val_th, n_th):
        if len(X) != len(y):
            print("Error")
            return None

        tp = 0  # true positives
        fp = 0  # false positives
        fn = 0  # false negatives
        for i in range(len(y)):
            example = X[i]
            label = y[i]
            example = (example >= val_th).astype(int)
            if np.count_nonzero(example) >= n_th:
                if label:
                    tp += 1
                else:
                    fp += 1
            elif label:
                    fn += 1

        if tp > 0:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
        else:
            precision = 0
            recall = 0

        return precision, recall

    @staticmethod
    def precision_recall_curve(X, y):

        v_ths = []
        n_ths = []
        precision = []
        recall = []
        number_thresholds = range(1, 300, 3)
        value_thresholds = np.arange(0.1, 1.81, 0.01)
        for n_th in number_thresholds:
            for v_th in value_thresholds:
                p, r = ModelThresholdBinary.evaluate(X, y, v_th, n_th)
                if p != 0 and r != 0:
                    v_ths.append(v_th)
                    n_ths.append(n_th)
                    precision.append(p)
                    recall.append(r)

        return v_ths, n_ths, precision, recall
