""" per discriminare, viene applicata una soglia alla somma delle radici quadrate effettuate su ognuno degli 8000 numeri del tensore  """
import numpy as np

class ModelBaseline:

    def __init__(self, parameters):
        self.threshold = -1
        self.max_sum = 1
        if "t" in parameters: # ha senso solo in caso di input non binarizzato, perché si usa per binarizzare
            self.threshold = float(parameters["t"])
        if "s" in parameters: # somma che dipende dalle statistiche
            self.max_sum = float(parameters["s"])

    def predict(self, X_test):
        predictions = []
        if self.threshold > 0:
            X_test = (X_test >= self.threshold).astype(bool)
        for i in range(len(X_test)):
            example = X_test[i]
            pred = np.sum(example) / self.max_sum
            if pred > 1:
                pred = 1
            predictions.append(pred)
        return predictions


def get_model(parameters):
    return ModelBaseline(parameters)

''' buoni modi per chiamare questo algoritmo '''
# dataset originale:
# baseline?s=55
# dataset originale, binarizzandolo:
# baseline?t=0.0065,s=8000
# dataset con un livello di average pooling:
# baseline?s=55
# dataset con un livello di average pooling, binarizzandolo:
# baseline?t=0.0065,s=1000
# dataset con due livelli di average pooling:
# baseline?s=55
# dataset con un livello di average pooling, binarizzandolo:
# baseline?t=0.0065,s=125

# se si vuole lanciare su un dataset già binarizzato, non passare il parametro t
