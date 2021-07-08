""" per discriminare, viene applicata una soglia alla somma delle radici quadrate effettuate su ognuno degli 8000 numeri del tensore  """
import numpy as np

# TODO da provare
class ModelBaselineSQRT:

    def __init__(self, parameters):
        self.max_sum = 1
        if "s" in parameters: # somma che dipende dalle statistiche
            self.max_sum = float(parameters["s"])

    def predict(self, X_test):
        predictions = []
        X_test = np.sqrt(X_test)
        for i in range(len(X_test)):
            example = X_test[i]
            pred = np.sum(example) / self.max_sum
            if pred > 1:
                pred = 1
            predictions.append(pred)
        return predictions


def get_model(parameters):
    return ModelBaselineSQRT(parameters)

''' buoni modi per chiamare questo algoritmo '''
# dataset originale:
# baseline_sqrt?s=300
# dataset con un livello di average pooling:
# baseline_sqrt?s=50
# dataset con due livelli di average pooling:
# baseline_sqrt?s=8

# binarizzare o usare dataset non binarizzati non ha senso,
# in quanto cambierebbero solo le soglie nel primo caso e nel secondo si farebbero inutilmente le radici
