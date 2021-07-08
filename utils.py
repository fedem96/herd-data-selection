""" raccolta di funzioni varie che vengono usate dagli altri script """

import os
import psutil

from glob import glob

import random

import numpy as np
import h5py


def memory_usage():
    process = psutil.Process(os.getpid())
    mem_MiB = process.memory_info().rss/1024**2
    return str(int(mem_MiB)) + "MiB"


# data una lista, la funzione crea un dizionario le cui chiavi sono elementi della lista, e i valori sono il numero di occorrenze dell'elemento
def count_occurences(my_list):
    dictionary = {}
    for element in my_list:
        str_element = str(element)
        if str_element in dictionary.keys():
            dictionary[str_element] += 1
        else:
            dictionary[str_element] = 1
    return dictionary


# funzione che legge i dati dai file in formato .npz
# dir: cartella contenente i file da leggere
# num_files: numero di file da leggere
def read_particles(directory, num_files=0, log_file=None, SEED=None):

    files = glob(directory)

    files.sort()
    #print("list of files:")
    #print(files)

    # np array che conterra' il dataset
    X = np.array([])
    # lista che conterra' le etichette della bonta' degli esempi
    y_good = []

    if num_files > 0:
        files = files[:num_files]
    for pf in files:
        print("reading file " + str(pf))
        if log_file is not None:
            log_file.write("reading file " + str(pf) + "\n")

        # leggo contenuto del file
        file_content = np.load(pf)

        # leggo gli esempi del file
        hits = file_content['hits']
        if len(X) > 0:
            # se X non e' vuoto, aggiungo ad X gli esempi
            X = np.concatenate([X, hits])
        else:
            # se X e' vuoto, lo inizializzo
            X = hits

        # leggo le etichette relative alla bonta' dei dati
        good = file_content['good']

        # le aggiungo alla lista
        y_good.extend(good)

    X = X.reshape((len(X), 20, 20, 20, 1))
    y_good = np.array(y_good)

    if SEED is not None:
        np.random.seed(SEED)
        np.random.shuffle(X)
        np.random.seed(SEED)
        np.random.shuffle(y_good)

    return X, y_good


# funzione che legge i dati da un file in formato .hdf5
def read_particles_hdf5(file, num_examples=0, log_file=None, must_rebalance=False, rebalance_seed=None): #, rebalance_positive_fraction=0.5):

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


# elimina esempi la cui etichetta e' piu' ricorrente, selezionandoli casualmente, fino a ribilanciare il dataset
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

#def prepro
