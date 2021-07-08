""" script che permette di calcolare alcune statistiche sui dati """

import os, sys
import numpy as np
import pickle

import matplotlib
matplotlib.use("Agg")   # necessario per salvare immagini di matplotlib quando lo script gira su un server senza GUI
import matplotlib.pyplot as plt

from utils import rebalance, memory_usage, read_particles_hdf5


def print_stat(name, value, output_file=None):
    print("\t" + name + str(value))
    if output_file is not None:
        output_file.write("\t" + name + str(value) + "\n")



def print_statistics(name, dataset, axis=(1, 2, 3, 4), output_file=None, calc_max_min=True):

    dataset_length = len(dataset)
    print(name + " - " + str(dataset_length) + " examples")
    if output_file is not None:
        output_file.write(name + " - " + str(dataset_length) + " examples\n")

    sums = np.sum(dataset, axis=axis)

    if calc_max_min:
        print_stat("maximum: ", np.max(sums), output_file)
    print_stat("mean:    ", np.mean(sums), output_file)
    #print_stat("median:  ", np.median(sums), output_file)
    if calc_max_min:
        print_stat("minimum: ", np.min(sums), output_file)
    print_stat("variance:", np.var(sums), output_file)

    print("memory used: " + memory_usage())


def print_all_statistics(X, name, output_dir_plots, output_file=None):
    print_statistics(name + " - sum - all examples ", X, output_file=output_file)

    X_good = [X[i] for i in range(len(X)) if y[i]]
    print_statistics(name + " - sum - good examples", X_good, output_file=output_file)
    plot_sums_distribution(X_good, output_dir_plots + "/good")
    X_good = None

    X_no_good = [X[i] for i in range(len(X)) if not y[i]]
    print_statistics(name + " - sum - bad examples", X_no_good, output_file=output_file)
    plot_sums_distribution(X_no_good, output_dir_plots + "/no_good")
    X_no_good = None

    plt.legend(("good", "no good"))
    plt.title(output_dir_plots)
    plt.savefig(output_dir_plots + "/distribution.png")
    plt.clf()

    if binarized:
        print_statistics(name + " - single element:", X, 4, output_file, calc_max_min=False)
    else:
        print_statistics(name + " - single element:", X, 4, output_file)

def plot_sums_distribution(dataset, output_dir, axis=(1, 2, 3, 4)):
    # TODO queste somme andrebbero fatte una volta sola, vanno riorganizzati meglio un po' tutte le funzioni
    sums = np.sum(dataset, axis=axis).astype(int)
    sums = sorted(sums)
    values = [sums[0]]
    counts = [1]
    for i in range(1, len(sums)):
        if sums[i] == values[-1]:
            counts[-1] += 1
        else:
            values.append(sums[i])
            counts.append(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pickle.dump(values, open(output_dir + "/values.p", "wb"))
    pickle.dump(counts, open(output_dir + "/counts.p", "wb"))

    plt.plot(values, counts)


if __name__ == "__main__":
    # TODO pulire codice

    ''' Definizione dei valori di default, possono essere modificati calcolare le statistiche su dati diversi '''
    particles = "electrons"
    avg_pooled = 0
    binarized = False
    b_threshold = 0.0065
    seed = 0
    rebalance_seed = 0
    num_examples = 0    # numero di esempi da leggere, numero <= 0 per leggere tutti gli esempi


    argv_i = 1
    while sys.argv[argv_i][0] == "-":
        arg = sys.argv[argv_i][1]
        # print(arg)
        if arg == "a":
            avg_pooled = 1
            if len(sys.argv[argv_i]) >= 3 and sys.argv[argv_i][2] == "a":
                avg_pooled += 1
        elif arg == "b":
            binarized = True
            argv_i += 1
            if argv_i < len(sys.argv):
                try:
                    b_threshold = float(sys.argv[argv_i])
                except ValueError:
                    print("error: invalid binarization threshold")
                    exit(2)
            else:
                print("error: binarization threshold not specified")
                exit(3)
        elif arg == "e":
            particles = "electrons"
        elif arg == "n":
            argv_i += 1
            if argv_i < len(sys.argv):
                try:
                    num_examples = int(sys.argv[argv_i])
                except ValueError:
                    print("error: invalid number of examples")
                    exit(2)
            else:
                print("error: number of examples not specified")
                exit(3)
        elif arg == "p":
            particles = "protons"
        else:
            print("error: unknown parameter " + sys.argv[argv_i])
            exit(4)
        argv_i += 1

        if argv_i == len(sys.argv):
            break



    ''' Parametri auto-definiti, non devono essere modificati '''
    str_dataset_type = ""
    if avg_pooled == 1:
        str_dataset_type = "(a)"
    elif avg_pooled == 2:
        str_dataset_type = "(a)(a)"
    if binarized:
        str_dataset_type += "(b" + str(b_threshold) + ")"
    input_file = "../infn/h5converted/" + particles + "_sphere/data/dataset-seed_" + str(seed) + str_dataset_type + ".hdf5"
    #input_file = "../infn/h5converted/" + particles + "_sphere/data/dataset-all_files.hdf5"
    if num_examples <= 0:
        str_num_examples = ""
    else:
        str_num_examples = "-" + str(num_examples) + "_examples"

    output_directory = "statistics/" + particles + str_dataset_type + str_num_examples

    if num_examples <= 0:
        str_num_examples = "all"
    else:
        str_num_examples = str(num_examples)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = open(output_directory + "/stat_" + str_num_examples + "_examples.txt", 'w')
    output_dir_plots = output_directory + "/dist_" + str_num_examples + "_examples"

    ''' Caricamento del dataset '''
    X, y = read_particles_hdf5(input_file, num_examples=num_examples, log_file=log_file)
    if binarized:
        X = X.astype(bool) # non testato
    X = X.astype('float16')

    ''' Calcolo statistiche '''
    # senza ribilanciamento
    print_all_statistics(X, "Original", output_dir_plots + "-no_rebalance-simple_sums", log_file)
    if not binarized:
        X_sqrt = np.sqrt(X)
        print_all_statistics(X_sqrt, "Square root", output_dir_plots + "-no_rebalance-sqrt_sums", log_file)
        X_sqrt = None

    # con ribilanciamento (stesso numero di etichette positive e negative)
    X, y = rebalance(X, y, rebalance_seed)

    print_all_statistics(X, "Original with rebalance", output_dir_plots + "-with_rebalance-simple_sums", log_file)
    if not binarized:
        X_sqrt = np.sqrt(X)
        X = None
        print_all_statistics(X_sqrt, "Square root with rebalance", output_dir_plots + "-with_rebalance-sqrt_sums", log_file)

    log_file.close()

