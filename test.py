""" script che permette di testare e confrontare diversi modelli """

import datetime
import importlib
import os
import sys
import pickle

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score

import matplotlib
matplotlib.use('Agg')   # necessario per salvare immagini di matplotlib quando lo script gira su un server senza GUI
import matplotlib.pyplot as plt

from utils import read_particles_hdf5

import csv

#models_names = []  # lista di cartelle dei modelli (solo nome cartella dentro models, non l'intero percorso)
particles = "protons"
seed = 0
num_examples = 0    # numero di elementi da leggere, numero <= 0 per leggerli tutti
avg_pooled = 0
binarized = False
b_threshold = 0.5
gpu = "0"
# rebalance = False  # indica se si vuole bilanciare i dati, deve essere lasciato su False perché la percentuale di dati buoni è simile a quella vera
# rebalance_seed = 0
#rebalance_fraction = 0.5
mode_final_test = False # se impostata su False, viene usato il validation set; se impostata su True, viene usato il test set
comparison = False

if len(sys.argv) == 1:
    print("error: no model specified")
    exit(1)

argv_i = 1
while sys.argv[argv_i][0] == "-":
    arg = sys.argv[argv_i][1]
    #print(arg)
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
    elif arg == "c":
        comparison = True
    elif arg == "e":
        particles = "electrons"
    elif arg == "g":
        argv_i += 1
        if argv_i < len(sys.argv):
            gpu = sys.argv[argv_i]
        else:
            print("error: gpu not specified")
            exit(3)
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
    # ribilanciamento disabilitato perché non deve essere fatto
    # elif arg == "r":
    #     rebalance = True
    #     argv_i += 1
    #     if argv_i < len(sys.argv):
    #         try:
    #             rebalance_fraction = float(sys.argv[argv_i])
    #         except ValueError:
    #             print("error: invalid rebalance fraction")
    #             exit(2)
    #     else:
    #         print("error: rebalance fraction not specified")
    #         exit(3)
    elif arg == "t":
        mode_final_test = True
    else:
        print("error: unknown parameter " + sys.argv[argv_i])
        exit(4)
    argv_i += 1
    if argv_i == len(sys.argv):
        print("error: no model specified")
        exit(1)

models_full_names = []
dict_datasets = {}
while argv_i < len(sys.argv):
    options = ""
    while sys.argv[argv_i][0] == "-":
        arg = sys.argv[argv_i][1]
        #print(arg)
        if arg == "a":
            options = "a" + options
            if len(sys.argv[argv_i]) >= 3 and sys.argv[argv_i][2] == "a":
                options = "a" + options
        elif arg == "b":
            options += "b "
            argv_i += 1
            if argv_i < len(sys.argv):
                try:
                    b_th = float(sys.argv[argv_i])
                    options += sys.argv[argv_i]
                except ValueError:
                    print("error: invalid binarization threshold")
                    exit(2)
            else:
                print("error: binarization threshold not specified")
                exit(3)
        else:
            print("error: unknown parameter " + sys.argv[argv_i])
            exit(4)
        argv_i += 1
        if argv_i == len(sys.argv):
            print("error: no model specified after dataset options")
            exit(1)

    while argv_i < len(sys.argv) and sys.argv[argv_i][0] != "-":
        models_full_names.append(sys.argv[argv_i])
        dict_datasets[sys.argv[argv_i]] = options
        argv_i += 1

models_names = [m for m in models_full_names]

if len(models_names) == 1:
    comparison = False

models_parameters = []
for i in range(len(models_names)):
    params = {}
    q_mark_idx = models_names[i].find("?")
    if q_mark_idx != -1: # se ho specificato almeno un parametro per questo modello
        assignments = models_names[i][q_mark_idx+1:].split(",")
        models_names[i] = models_names[i][:q_mark_idx]
        for assignment in assignments:
            key, val = assignment.split("=")
            params[key] = val
    models_parameters.append(params)

print("average pooled: " + str(avg_pooled))
print("binarized: " + str(binarized))
print("binarization threshold : " + str(b_threshold))
print("gpu: " + gpu)
print("models names: " + str(models_names))
print("num examples: " + str(num_examples))
print("particles: " + particles)
# print("rebalance: " + str(rebalance))
#print("rebalance fraction: " + str(rebalance_fraction))
# print("rebalance seed: " + str(rebalance_seed))
print("seed: " + str(seed))
print("test used: " + str(mode_final_test))


if mode_final_test:
    str_mode = "test"
else:
    str_mode = "validation"

''' Caricamento del dataset '''
# percorso del file da cui leggere i dati
str_dataset_type = ""
if avg_pooled == 1:
    str_dataset_type = "(a)"
elif avg_pooled == 2:
    str_dataset_type = "(a)(a)"
if binarized:
    str_dataset_type += "(b" + str(b_threshold) + ")"
data_directory = "../infn/h5converted/" + particles + "_sphere/data/dataset-seed_" + str(seed) + str_dataset_type + ".hdf5"
X, y = read_particles_hdf5(data_directory, num_examples) # seed
dataset_length = len(X)

''' Divisione in Train, Validation e Test'''
# Train:      64%
# Validation: 16%
# Test:       20%

beginValidateIndex = int(dataset_length * 64 / 100)
beginTestIndex = int(dataset_length * 80 / 100)

if mode_final_test:
    # carico il test set
    X_test = X[beginTestIndex:]
    y_test = y[beginTestIndex:]
else:
    # carico il validation set
    X_test = X[beginValidateIndex:beginTestIndex]
    y_test = y[beginValidateIndex:beginTestIndex]

X = None
original_shape = X_test.shape

if num_examples <= 0:
    str_examples = ""  # se non scrivo niente è sottointeso che si considerano tutti gli esempi
else:
    str_examples = "-" + str(num_examples) + "_examples"

''' Inizio il test '''
for model_name in models_names:
    if model_name.startswith("conv"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use, usually either "0" or "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        from keras.models import load_model
        break

plt.figure(1)
plt.title("PR curve")

plt.figure(2)
plt.title("ROC curve")

#str_rebalance = "-rebalance_s" + str(rebalance_seed)
str_info = ""
for i in range(len(models_names)):
    model_name = models_names[i]
    model_full_name = models_full_names[i]
    model_parameters = models_parameters[i]

    if i > 0 and dict_datasets[model_full_name] != dict_datasets[models_full_names[i-1]]:
        options = dict_datasets[model_full_name]

        str_dataset_type = ""
        if options.count("a") == 1:
            str_dataset_type = "(a)"
        elif options.count("a") == 2:
            str_dataset_type = "(a)(a)"
        if options.count("b") == 1:
            str_b_threshold = options.split(" ")[1]
            str_dataset_type += "(b" + str_b_threshold + ")"

        data_directory = "../infn/h5converted/" + particles + "_sphere/data/dataset-seed_" + str(seed) + str_dataset_type + ".hdf5"
        X, y = read_particles_hdf5(data_directory, num_examples)
        if mode_final_test:
            # carico il test set
            X_test = X[beginTestIndex:]
            y_test = y[beginTestIndex:]
        else:
            # carico il validation set
            X_test = X[beginValidateIndex:beginTestIndex]
            y_test = y[beginValidateIndex:beginTestIndex]

        X = None
        original_shape = X_test.shape

    print("begin testing " + model_name)

    ''' Parametri auto-definiti, non devono essere modificati '''
    model_type = ""
    if model_name.startswith('baseline'):
        model_type = 'baseline'
    elif model_name.startswith('conv'):
        model_type = 'conv'
    elif model_name.startswith('perceptron'):
        model_type = 'perceptron'
    elif model_name.startswith('svm'):
        model_type = 'svm'
    else:
        print("tipo di modello sconosciuto")
        exit(1)


    ''' Caricamento del modello '''
    model_path = "models." + model_name + ".model"
    model_module = importlib.import_module(model_path)
    try:
        fit_settings = model_module.get_fit_settings()
    except AttributeError:
        fit_settings = {}

    fit_settings.update(model_parameters)

    # cartella dalla quale viene letto il modello da testare
    model_directory = "experiments/" + particles + "-seed_" + str(seed) + "/" \
                      + str_dataset_type + model_name + str_examples
    for parameter, value in fit_settings.items():
        model_directory += "-" + parameter + "_" + str(value)

    model_file = model_directory + "/model.hdf5"

    if model_type == 'conv':
        model = load_model(model_file)
        model.summary()
    elif model_type == 'baseline':
        model = model_module.get_model(model_parameters)
    else:
        model = pickle.load(open(model_file, 'rb'))

    ''' Valutazione del modello '''
    if model_type == 'conv':
        X_test = X_test.reshape(original_shape)
        evaluation = model.evaluate(X_test, y_test)
        print(evaluation)
    elif model_type == 'baseline':
        X_test = X_test.reshape(original_shape)
    elif model_type == 'perceptron':
        X_test = X_test.reshape(len(X_test), -1)

    if model_type != 'perceptron':
        y_pred = model.predict(X_test)
    else:
        y_pred = np.transpose(model.predict_proba(X_test))[1]

    plt.figure(1)
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_pred)
    if len(recalls) > 2:
        recalls = recalls[:-1]
        precisions = precisions[:-1]
    area_under_pr_curve = auc(recalls, precisions)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        print("dir created")
    i_OP = -1
    # (come OP cerco prima il punto con recall più alta, e sia precision che recall almeno 90%) tolto questo criterio
    str_csv = model_directory + "/" + str_mode + "-thresholds.csv"
    if not os.path.exists(str_csv):
        with open(str_csv, 'wt') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';')
            csv_writer.writerow(['threshold', 'precision', 'recall'])
    #    for i in range(len(recalls) - 1):
    #        csv_writer.writerow([thresholds_pr[i], precisions[i], recalls[i]])
    #        if precisions[i] >= 0.9 and (i_OP == -1 or recalls[i] > recalls[i_OP]
    #                                     or (recalls[i] == recalls[i_OP] and precisions[i] > precisions[i_OP])):
    #            i_OP = i

    # cerco un OP che soddisfi la soglia di 90p/90r, minimizzando una certa distanza
    if i_OP == -1:
        min_distance = 3
        for j in range(len(recalls)):
            if precisions[j] >= 0.9 and recalls[j] >= 0.9:
                # calcolo distanza ~manhattan tra il punto (recalls[i], precisions[i]) ed il punto (1,1),
                # dando importanza doppia alla recall rispetto alla precision
                distance = 3-2*recalls[j]-precisions[j]
                if distance < min_distance or i_OP == -1:
                    i_OP = j
                    min_distance = distance

    # se prima non ho trovato un OP soddisfacente, cerco quello che minimizza una certa distanza in generale
    if i_OP == -1:
        min_distance = 3
        for j in range(len(recalls)):
            # calcolo distanza ~manhattan tra il punto (recalls[i], precisions[i]) ed il punto (1,1),
            # dando importanza doppia alla recall rispetto alla precision
            distance = 3-2*recalls[j]-precisions[j]
            if distance < min_distance or i_OP == -1:
                i_OP = j
                min_distance = distance

    if i_OP != -1:
        # se ho trovato un OP
        str_operating_point = ', OP = ({:.3f}, {:.3f})'.format(precisions[i_OP], recalls[i_OP])     # , threshold_pr_op)
    else:
        # OP sconosciuto
        str_operating_point = ', OP = ?'

    plt.plot(recalls, precisions, label=model_name + ': AUC = {:.3f}'.format(area_under_pr_curve) + str_operating_point)

    plt.figure(2)
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred)
    area_under_roc_curve = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=model_name+': AUC = {:.3f}'.format(area_under_roc_curve))

    dict_metrics = {"OPp": precisions[i_OP], "OPr": recalls[i_OP], "AUC": area_under_roc_curve, "AUCpr": area_under_pr_curve}
    pickle.dump(dict_metrics, open(model_directory + "/metrics.p", "wb"))
    print("metrics saved in " + model_directory + "/metrics.p")
    print("end testing " + model_name)

    str_info += data_directory + ";" + model_full_name + "\n"

    if (not comparison) or (i == len(models_names)-1):

        ''' Salvataggio dei risultati '''
        if not comparison:
            output_directory = model_directory
        else:
            timestamp = str(datetime.datetime.now())
            timestamp = timestamp.replace(" ", "_")
            timestamp = timestamp[:timestamp.find(".")]
            output_directory = "experiments/" + particles + "-seed_" + str(seed) + "/comparison_" + timestamp

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if comparison:
            file_info = open(output_directory + "/" + str_mode + "_info.txt", "at")
            file_info.write(str_info)

        plt.figure(1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        ax = plt.subplot(111)
        if particles == 'electrons':
            ax.set_xlim(0.7, 1.02)
            ax.set_ylim(0.7, 1.02)
            ax.set_xticks([0.7, 0.8, 0.9, 1.0])
            ax.set_yticks([0.7, 0.8, 0.9, 1.0])
        else:
            ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(output_directory + "/pr-" + str_mode + ".png")
        plt.clf()

        plt.figure(2)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        ax = plt.subplot(111)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(output_directory + "/roc-" + str_mode + ".png")
        plt.clf()

        print("risultati riportati in: " + output_directory)


print("test concluso")
print("\n")
