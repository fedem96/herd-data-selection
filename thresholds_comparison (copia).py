import importlib
import os
import sys
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
#from utils import read_particles_hdf5

gpu = "0"
particles = ""
num_examples = 0
avg_pooled = 0
seed = 0
models_names = []
metric = "AUCpr"

if len(sys.argv) == 1:
    print("error: no threshold specified")
    exit(1)

argv_i = 1
cli_params = ""
while sys.argv[argv_i][0] == "-":
    arg = sys.argv[argv_i][1]
    #print(arg)
    if arg == "a":
        avg_pooled = 1
        if len(sys.argv[argv_i]) >= 3 and sys.argv[argv_i][2] == "a":
            avg_pooled += 1
            cli_params += "-aa "
        else:
            cli_params += "-a "
    elif arg == "e":
        particles = "electrons"
        cli_params += "-e "
    elif arg == "g":
        argv_i += 1
        if argv_i < len(sys.argv):
            gpu = sys.argv[argv_i]
            cli_params += "-g " + gpu + " "
        else:
            print("error: gpu not specified")
            exit(3)
    elif arg == "n":
        argv_i += 1
        if argv_i < len(sys.argv):
            try:
                num_examples = int(sys.argv[argv_i])
                cli_params += "-n " + sys.argv[argv_i] + " "
            except ValueError:
                print("error: invalid number of examples")
                exit(2)
        else:
            print("error: number of examples not specified")
            exit(3)
    elif arg == "p":
        particles = "protons"
        cli_params += "-p "
    argv_i += 1
    if argv_i == len(sys.argv):
        print("error: no threshold specified")
        exit(1)

if particles == "":
    print("error: no particles specified")
    exit(5)
thresholds = [float(th) for th in sys.argv[argv_i:]]

if num_examples <= 0:
    str_examples = ""  # se non scrivo niente è sottointeso che si considerano tutti gli esempi
else:
    str_examples = "-" + str(num_examples) + "_examples"

str_dataset_type = ""
if avg_pooled == 1:
    str_dataset_type = "(a)"
elif avg_pooled == 2:
    str_dataset_type = "(a)(a)"

ths_dir = "thresholds_" + metric + "/" + particles + str_dataset_type
if not os.path.exists(ths_dir):
    os.makedirs(ths_dir)

if avg_pooled == 0:
    models_full_names = ["baseline?s=8000", "perceptron-2018_10_04-v0", "conv2d-2018_09_17-v0?epochs=10", "conv3d-2018_08_07-v0?epochs=10"]
    tensor_side = 20
elif avg_pooled == 1:
    models_full_names = ["baseline?s=1000", "perceptron-2018_10_04-v0", "conv2d-2018_09_17-v2?epochs=10",
                         "conv3d-2018_09_21-v0?epochs=10"]
    tensor_side = 10
else:
    print("operation not supported yet")
    exit(5)

models_names = [m for m in models_full_names]
file_evaluations = ths_dir + "/evaluations_" + str(tensor_side) + str_examples + ".p"
colors = ["b", "y", "r", "g"]

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

evaluations = {}
if os.path.isfile(file_evaluations):
    evaluations = pickle.load(open(file_evaluations, "rb"))

dataset_name = "dataset-seed_" + str(seed) + str_dataset_type
dataset_file_prefix = "../infn/h5converted/" + particles + "_sphere/data/" + dataset_name
dataset_file = dataset_file_prefix + ".hdf5"
#X_original, y = read_particles_hdf5(dataset_file, num_examples)


for th in thresholds:
    ds_th = dataset_file_prefix + "(b" + str(th) + ").hdf5"
    if not os.path.isfile(ds_th):
        print("binarizing dataset_file with threshold " + str(th))
        dataset_name_cli = dataset_name.replace("(", "\(").replace(")", "\)")
        exit_status = os.system("python binarize.py " + particles + " " + dataset_name_cli + ".hdf5 " + str(th))
        if exit_status != 0:
            print("error: " + str(exit_status))
            exit(exit_status)

training_models = ""
testing_models = ""
# cerco quali modelli devo ancora allenare e/o testare, sui dati non binarizzati
for i in range(len(models_names)):

    model_name = models_names[i]
    model_parameters = models_parameters[i]

    model_path = "models." + model_name + ".model"
    model_module = importlib.import_module(model_path)
    #model = model_module.get_model()
    try:
        fit_settings = model_module.get_fit_settings()
    except AttributeError:
        fit_settings = {}

    fit_settings.update(model_parameters)

    # cartella nella quale sara' salvato il modello
    model_exp_dir = "experiments/" + particles + "-seed_" + str(seed) + "/" \
                       + str_dataset_type + model_name + str_examples
    for parameter, value in fit_settings.items():
        model_exp_dir += "-" + parameter + "_" + str(value)

    if model_name not in evaluations:

        if (not model_name.startswith("baseline")) and (not os.path.exists(model_exp_dir)):
            training_models += " " + models_full_names[i]
            print(model_name + " needs to be trained on real values data")

        if not os.path.isfile(model_exp_dir + "/metrics.p"):
            testing_models += " " + models_full_names[i]
            print(model_name + " needs to be tested on real values data")

# alleno quei modelli che non sono stati allenati sui dati a valori reali
if training_models != "":
    print("trainings on real values data starting...")
    exit_status = os.system("python train.py " + cli_params + training_models)
    if exit_status != 0:
        print("error: " + str(exit_status))
        exit(exit_status)
# testo quei modelli che non sono stati testati sui dati a valori reali
if testing_models != "":
    print("tests on real values data starting...")
    exit_status = os.system("python test.py " + cli_params + testing_models)
    if exit_status != 0:
        print("error: " + str(exit_status))
        exit(exit_status)

# per ogni modello, disegno una riga orizzontale tratteggiata che rappresenta le sue prestazioni con dati a valori reali
for i in range(len(models_names)):
    model_name = models_names[i]
    color = colors[i]
    model_parameters = models_parameters[i]

    model_path = "models." + model_name + ".model"
    model_module = importlib.import_module(model_path)
    #model = model_module.get_model()
    try:
        fit_settings = model_module.get_fit_settings()
    except AttributeError:
        fit_settings = {}

    fit_settings.update(model_parameters)

    # cartella nella quale sara' salvato il modello
    model_exp_dir = "experiments/" + particles + "-seed_" + str(seed) + "/" \
                    + str_dataset_type + model_name + str_examples
    for parameter, value in fit_settings.items():
        model_exp_dir += "-" + parameter + "_" + str(value)

    if model_name not in evaluations:
        metrics = pickle.load(open(model_exp_dir + "/metrics.p", "rb"))
        evaluations[model_name] = [metrics[metric], {}]

    pickle.dump(evaluations, open(file_evaluations, "wb"))
    #plt.axhline(y=evaluations[model_name][0], color=color, linestyle='--')
    print("plotting dotted line for model " + model_name + ", val = " + str(evaluations[model_name][0]))
    plt.semilogx(thresholds, [evaluations[model_name][0] for k in range(len(thresholds))], color + '--', basex=2)


# cerco quali modelli devo ancora allenare e/o testare, sui dati non binarizzati
for th in thresholds:
    ds_th = dataset_file_prefix + "(b" + str(th) + ").hdf5"
    training_models = ""
    testing_models = ""

    for i in range(len(models_names)):
        model_name = models_names[i]
        model_parameters = models_parameters[i]

        model_path = "models." + model_name + ".model"
        model_module = importlib.import_module(model_path)
        #model = model_module.get_model()
        try:
            fit_settings = model_module.get_fit_settings()
        except AttributeError:
            fit_settings = {}

        fit_settings.update(model_parameters)

        # cartella nella quale sara' salvato il modello
        model_exp_dir = "experiments/" + particles + "-seed_" + str(seed) + "/" \
                        + str_dataset_type + "(b" +  str(th) + ")" + model_name + str_examples
        for parameter, value in fit_settings.items():
            model_exp_dir += "-" + parameter + "_" + str(value)

        if th not in evaluations[model_name][1]:

            if (not model_name.startswith("baseline")) and (not os.path.isfile(model_exp_dir + "/model.hdf5")):
                training_models += " " + models_full_names[i]

            if not os.path.isfile(model_exp_dir + "/metrics.p"):
                testing_models += " " + models_full_names[i]

    # alleno quei modelli che non sono stati allenati sui dati con soglia th
    if training_models != "":
        exit_status = os.system("python train.py " + cli_params + "-b " + str(th) + training_models)
        if exit_status != 0:
            print("error: " + str(exit_status))
            exit(exit_status)
    # testo quei modelli che non sono stati testati sui dati con soglia th
    if testing_models != "":
        exit_status = os.system("python test.py " + cli_params + "-b " + str(th) + testing_models)
        if exit_status != 0:
            print("error: " + str(exit_status))
            exit(exit_status)

    # vado a vedere le metriche dei modelli per la soglia th
    for i in range(len(models_names)):
        model_name = models_names[i]
        model_parameters = models_parameters[i]

        model_path = "models." + model_name + ".model"
        model_module = importlib.import_module(model_path)
        #model = model_module.get_model()
        try:
            fit_settings = model_module.get_fit_settings()
        except AttributeError:
            fit_settings = {}

        fit_settings.update(model_parameters)

        # cartella nella quale sara' salvato il modello
        model_exp_dir = "experiments/" + particles + "-seed_" + str(seed) + "/" \
                        + str_dataset_type + "(b" +  str(th) + ")" + model_name + str_examples
        for parameter, value in fit_settings.items():
            model_exp_dir += "-" + parameter + "_" + str(value)

        if th not in evaluations[model_name][1]:
            metrics = pickle.load(open(model_exp_dir + "/metrics.p", "rb"))
            evaluations[model_name][1][th] = metrics[metric]
            pickle.dump(evaluations, open(file_evaluations, "wb"))


# per ogni modello vado a disegnare il grafico della metrica in funzione della soglia
for i in range(len(models_names)):
    model_name = models_names[i]
    color = colors[i]
    model_parameters = models_parameters[i]
    recalls = []
    for th in thresholds:
        recalls.append(evaluations[model_name][1][th])

    print("plotting plot for model " + model_name + ", vals = " + str(recalls))
    minus_idx = model_name.find("-")
    str_label = model_name
    if minus_idx != -1:
        str_label = model_name[:minus_idx]
    plt.semilogx(thresholds, recalls, color + "-", label=str_label,basex=2)


plt.xlabel('Threshold')
plt.ylabel(metric)
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.82, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(ths_dir + "/evaluations_" + str(tensor_side) + str_examples + ".png")
