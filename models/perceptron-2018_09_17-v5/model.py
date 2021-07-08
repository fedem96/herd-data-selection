from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import Perceptron


def get_model():
    p = Perceptron(max_iter=100, eta0=1, verbose=2, penalty='l2', alpha=0.001)

    cccv = CalibratedClassifierCV(p, method='sigmoid')

    return cccv
