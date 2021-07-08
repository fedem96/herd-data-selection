""" per discriminare, viene applicata una soglia alla somma delle radici quadrate effettuate su ognuno degli 8000 numeri del tensore  """
import pickle
import random

import numpy as np

import h5py

def read_particles_hdf5(file, num_examples=0, log_file=None, must_rebalance=False, rebalance_seed=None, set='all'):

    if num_examples > 0:
        str_num_examples = str(num_examples)
    else:
        str_num_examples = "all"
    print("reading " + str_num_examples + " examples from file: " + str(file))

    if log_file is not None:
        log_file.write("reading " + str_num_examples + " examples from file: " + str(file) + "\n")

    f = h5py.File(file, 'r')
    X = f['examples']
    y = f['labels']

    if num_examples > 0:
        if must_rebalance:
            num_trues = 0
            num_falses = 0
            for i in range(len(X)):
                if y[i]:
                    num_trues += 1
                else:
                    num_falses += 1
                if num_trues >= int(num_examples/2) and num_falses >= int(num_examples/2) and num_trues + num_falses >= num_examples:
                    break
            X = X[:num_trues+num_falses]
            y = y[:num_trues+num_falses]
            X, y = rebalance(X, y, rebalance_seed)
        else:
            X = X[:num_examples]
            y = y[:num_examples]
    else:
        X = X[:]
        y = y[:]
        if must_rebalance:
            X, y = rebalance(X, y, rebalance_seed)

    return X, y

def rebalance(X, y, SEED=None):
    occurrences = count_occurences(y)
    true_occ = occurrences['True']
    false_occ = occurrences['False']
    if true_occ > false_occ:
        return remove_elements(X, y, True, true_occ-false_occ, SEED)
    elif false_occ > true_occ:
        return remove_elements(X, y, False, false_occ-true_occ, SEED)


# seleziona casualmente num_to_remove elementi che hanno label yVal, e li rimuove dal dataset
def remove_elements(X, y, yVal, num_to_remove, SEED=None):
    removing_indexes = [i for i in range(len(y)) if y[i] == yVal]
    if SEED is not None:
        r = random.Random()
        r.seed(SEED)
        r.shuffle(removing_indexes)
    removing_indexes = removing_indexes[:num_to_remove]
    X = np.delete(X, removing_indexes, axis=0)
    y = np.delete(y, removing_indexes)
    return X, y

def count_occurences(my_list):
    dictionary = {}
    for element in my_list:
        str_element = str(element)
        if str_element in dictionary.keys():
            dictionary[str_element] += 1
        else:
            dictionary[str_element] = 1
    return dictionary

class ModelBaseline:

    def __init__(self, file=None):
        self.threshold = 0.0002
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

if __name__ == "__main__":
    particles = "protons"   # electrons | protons

    num_examples = 70

    # random seed(s) da usare in lettura e/o split dei dati
    seed = 0
    # indicare se si vuole bilanciare i dati
    must_rebalance = True
    rebalance_seed = 0

    data_directory = "../../../infn/h5converted/" + particles + "_sphere/data/dataset-seed_" + str(seed) + ".hdf5"
    X, y = read_particles_hdf5(data_directory, num_examples, must_rebalance=must_rebalance, rebalance_seed=rebalance_seed)

    m = ModelBaseline()
    m.fit(X, y)

    m.predict(X)
