import os
import sys
import importlib

import numpy as np
from sklearn.utils import class_weight

from utils import read_particles_hdf5

gpu = "0"
particles = "electrons"
num_examples = 0
avg_pooled = 0
binarized = False
output_file = "random_tests_output.txt"
b_threshold = 0.0065

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
    elif arg == "o":
        argv_i += 1
        if argv_i < len(sys.argv):
            output_file = sys.argv[argv_i]
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

model_name = sys.argv[argv_i]

if model_name.startswith("conv"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # permette di far crescere dinamicamente la memoria usata dalla GPU
    sess = tf.Session(config=config)
    set_session(sess)

seed = 0
rebalance = False
rebalance_seed = 0


str_dataset_type = ""
if avg_pooled == 1:
    str_dataset_type = "(a)"
elif avg_pooled == 2:
    str_dataset_type = "(a)(a)"
if binarized:
    str_dataset_type += "(b" + str(b_threshold) + ")"

dataset_file = "../infn/h5converted/" + particles + "_sphere/data/dataset-seed_" + str(seed) + str_dataset_type + ".hdf5"
X_original, y = read_particles_hdf5(dataset_file, num_examples, must_rebalance=rebalance, rebalance_seed=rebalance_seed) # seed)


dataset_length = len(X_original)
beginValidateIndex = int(dataset_length * 64 / 100)
beginTestIndex = int(dataset_length * 80 / 100)

X_original_train = X_original[0:beginValidateIndex]
y_train = y[0:beginValidateIndex]

X_original_validate = X_original[beginValidateIndex:beginTestIndex]
y_validate = y[beginValidateIndex:beginTestIndex]

X_original_test = X_original[beginTestIndex:]
y_test = y[beginTestIndex:]

model_path = "models." + model_name + ".model"
model_module = importlib.import_module(model_path)
model = model_module.get_model()

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

print("model name: " + model_name)
# model.fit(X_original_train, y_train, validation_data=(X_original_validate, y_validate), epochs=7, batch_size=64, verbose=1, class_weight=class_weights)
# loss, accuracy = model.evaluate(X_original_test, y_test, verbose=1)
# print("without threshold, loss: " + str(loss) + ", accuracy: " + str(accuracy))


with open(output_file, "a") as file:
    file.write("dataset file: " + dataset_file + "\n")
    file.write("model name: " + model_name + "\n")

thresholds = [0.045, 0.035]

for i in range(len(thresholds)):

    #threshold = np.random.uniform(0.00001, 0.0065)
    threshold = thresholds[i]

    X = (X_original >= threshold).astype(bool)

    mean = np.mean(X)

    X_train = X[0:beginValidateIndex]
    X_validate = X[beginValidateIndex:beginTestIndex]
    #X_test = X[beginTestIndex:]

    model = model_module.get_model()
    model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=70, batch_size=64, verbose=1)
    loss, accuracy = model.evaluate(X_validate, y_validate, verbose=0)

    print("threshold: " + str(threshold) + ", mean: " + str(mean) + ", v_loss: " + str(loss) + ", v_accuracy: " + str(accuracy))
    with open(output_file, "a") as file:
        file.write("threshold: " + str(threshold) + ", mean: " + str(mean) + ", v_loss: " + str(loss) + ", v_accuracy: " + str(accuracy) + "\n")
