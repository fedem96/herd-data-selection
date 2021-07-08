""" script che permette di allenare un modello """

import pickle
import importlib
import os
import sys
from sklearn.utils import class_weight
import numpy as np

from utils import read_particles_hdf5

''' Definizione dei parametri di default, possono essere modificati per eseguire allenamenti diversi '''

#models_names = []  # lista di cartelle dei modelli (solo nome cartella dentro models, non l'intero percorso)
particles = "protons"
seed = 0
num_examples = 0    # numero di elementi da leggere, numero <= 0 per leggerli tutti
avg_pooled = 0
binarized = False
b_threshold = 0.5
gpu = "0"
# rebalance = True  # indica se si vuole bilanciare i dati
# rebalance_seed = 0
#rebalance_fraction = 0.5

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
    # ribilanciamento disabilitato
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
    else:
        print("error: unknown parameter " + sys.argv[argv_i])
        exit(4)
    argv_i += 1
    if argv_i == len(sys.argv):
        print("error: no model specified")
        exit(1)

models_names = sys.argv[argv_i:]

models_parameters = []
for i in range(len(models_names)):
    params = {}
    q_mark_idx = models_names[i].find("?")
    if q_mark_idx != -1: # se ho specificato almeno un parametro per questo modello
        assignments = models_names[i][q_mark_idx+1:].split(",")
        models_names[i] = models_names[i][:q_mark_idx]
        for assignment in assignments:
            key, val = assignment.split("=")
            if val.find(".") != -1:
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
            else:
                try:
                    params[key] = int(val)
                except ValueError:
                    params[key] = val

    models_parameters.append(params)

print("average pooled: " + str(avg_pooled))
print("binarized: " + str(binarized))
print("binarization threshold : " + str(b_threshold))
print("gpu: " + gpu)
print("models names: " + str(models_names))
print("num examples: " + str(num_examples))
print("particles: " + particles)
#print("rebalance: " + str(rebalance))
#print("rebalance fraction: " + str(rebalance_fraction))
# print("rebalance seed: " + str(rebalance_seed))
print("seed: " + str(seed))



''' Inizio allenamenti '''
for model_name in models_names:
    if model_name.startswith("conv"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use, usually either "0" or "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # permette di far crescere dinamicamente la memoria usata dalla GPU
        sess = tf.Session(config=config)
        set_session(sess)  # imposta questa sessione TensorFlow come la sessione di default per Keras.
        from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

        break


''' Caricamento del dataset '''
# percorso del file da cui leggere i dati
str_dataset_type = ""
if avg_pooled == 1:
    str_dataset_type = "(a)"
elif avg_pooled == 2:
    str_dataset_type = "(a)(a)"
if binarized:
    str_dataset_type += "(b" + str(b_threshold) + ")"
dataset_file = "../infn/h5converted/" + particles + "_sphere/data/dataset-seed_" + str(seed) + str_dataset_type + ".hdf5"

print("reading dataset")
#X, y = read_particles_hdf5(dataset_file, num_examples, must_rebalance=rebalance, rebalance_seed=rebalance_seed) # seed
X, y = read_particles_hdf5(dataset_file, num_examples)
dataset_length = len(X)


''' Divisione in Train, Validation e Test'''
# Train:      64%
# Validation: 16%
# Test:       20%

beginValidateIndex = int(dataset_length * 64 / 100)
beginTestIndex = int(dataset_length * 80 / 100)

X_train = X[0:beginValidateIndex]
y_train = y[0:beginValidateIndex]

X_validate = X[beginValidateIndex:beginTestIndex]
y_validate = y[beginValidateIndex:beginTestIndex]

train_original_shape = X_train.shape

X = None

#if rebalance:
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
# else:
#     class_weights = None

for i in range(len(models_names)):
    model_name = models_names[i]
    model_parameters = models_parameters[i]
    ''' Parametri auto-definiti, non devono essere modificati '''
    model_type = ""
    if model_name.startswith('baseline'):
        continue
    elif model_name.startswith('conv'):
        model_type = 'conv'
    elif model_name.startswith('perceptron'):
        model_type = 'perceptron'
    else:
        print("tipo di modello sconosciuto")
        exit(1)

    if num_examples <= 0:
        str_examples = ""   # se non scrivo niente è sottointeso che si considerano tutti gli esempi
    else:
        str_examples = "-" + str(num_examples) + "_examples"

    ''' Caricamento dinamico del modello '''
    model_path = "models." + model_name + ".model"
    model_module = importlib.import_module(model_path)
    model = model_module.get_model()
    try:
        fit_settings = model_module.get_fit_settings()
    except AttributeError:
        fit_settings = {}

    fit_settings.update(model_parameters)

    str_rebalance = ""
    #if rebalance:
    #str_rebalance = "-rebalance_s" + str(rebalance_seed)
    # cartella nella quale sara' salvato il modello
    model_saving_dir = "experiments/" + particles + "-seed_" + str(seed) + "/" \
                       + str_dataset_type + model_name + str_examples
    for parameter, value in fit_settings.items():
        model_saving_dir += "-" + parameter + "_" + str(value)

    model_saving_file = model_saving_dir + "/model.hdf5"

    ''' Allenamento '''
    if model_type == 'conv':
        tensorboard = TensorBoard(
            log_dir=model_saving_dir + '/logs'  # ,
            # write_graph=True,
            # write_grads=True,
            # histogram_freq=10,
            # write_images=True,
        )

        es = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=7,
                           verbose=1, mode='auto')

        mc = ModelCheckpoint(model_saving_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        fit_settings.update({'verbose': 1, 'validation_data': (X_validate, y_validate), 'callbacks': [tensorboard, es, mc], 'class_weight': class_weights})
        X_train = X_train.reshape(train_original_shape)
    elif model_type == 'perceptron':
        X_train = X_train.reshape(len(X_train), -1)

    print("begin training model " + model_name)
    model.fit(X_train, y_train, **fit_settings)
    print("end training model " + model_name)

    ''' Salvataggio del modello '''
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)

    if model_type == 'perceptron':
        pickle.dump(model, open(model_saving_file, 'wb'))

    # old
    # if model_type == 'baseline' or model_type == 'conv':
    #     model.save(model_saving_file)
    # else:
    #     pickle.dump(model, open(model_saving_file, 'wb'))

print("\n")
