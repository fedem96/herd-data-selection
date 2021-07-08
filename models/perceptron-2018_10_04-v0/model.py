from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron


def get_model():
    p = Perceptron(max_iter=30, eta0=1, verbose=2, class_weight='balanced')

    cccv = CalibratedClassifierCV(p, method='sigmoid')

    return cccv
