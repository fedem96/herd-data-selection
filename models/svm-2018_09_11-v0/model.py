from sklearn.svm import SVC


def get_model():
    return SVC(verbose=True, max_iter=10)
