""" script che permette di testare un modello """

import importlib, sys
import pickle

from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score

import matplotlib
matplotlib.use('Agg')   # necessario per salvare immagini di matplotlib quando lo script gira su un server senza GUI
import matplotlib.pyplot as plt

from utils import read_particles_hdf5

''' Definizione dei parametri, possono essere modificati per eseguire il test con modello e/o dati diversi '''
if len(sys.argv) > 2:
    particles = sys.argv[1]
    model_name = sys.argv[2]
    num_examples = 0
else:
    # nome della cartella in cui si trova il modello (non il percorso, solo cartella)
    #model_name = "conv2d-2018_07_17-v0"
    #model_name = "conv3d-2018_08_07-v0"
    model_name = "baseline_v2"
    #model_name = "baseline"

    # tipo di particelle
    particles = "electrons"   # electrons | protons
    # numero di elementi da leggere, numero <= 0 per leggerli tutti
    num_examples = 4000

# random seed(s) da usare in lettura e/o split dei dati
seed = 0
rebalance_seed = 0
# indicare se si vuole bilanciare i dati
must_rebalance = True
# se impostata su False, viene usato il validation set; se impostata su True, viene usato il test set
mode_final_test = False

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

if num_examples <= 0:
    str_examples = "all_examples"
else:
    str_examples = str(num_examples) + "_examples"


''' Caricamento del modello '''
model_path = "models." + model_name + ".model"
model_module = importlib.import_module(model_path)
try:
    fit_settings = model_module.get_fit_settings()
except AttributeError:
    fit_settings = {}

# percorso della cartella in cui leggere i dati
data_directory = "../infn/h5converted/" + particles + "_sphere/data/dataset-seed_" + str(seed) + ".hdf5"

str_rebalance = ""
if must_rebalance:
    str_rebalance = "-rebalance_s" + str(rebalance_seed)
# cartella dalla quale viene letto il modello da testare
model_directory = "experiments/" + particles + "-seed_" + str(seed) + str_rebalance + "/" + model_name + "-" + str_examples
for parameter, value in fit_settings.items():
    model_directory += "-" + parameter + "_" + str(value)

model_file = model_directory + "/model.hdf5"

if model_type == 'conv':
    from keras.models import load_model
    model = load_model(model_file)
elif model_type == 'baseline':
    model = model_module.get_model(model_file)
else:
    model = pickle.load(open(model_file, 'rb'))


''' Caricamento del dataset '''
# X, y = read_particles_hdf5(data_directory, num_examples, must_rebalance=must_rebalance, rebalance_seed=rebalance_seed) # seed
# dataset_length = len(X)


''' Divisione in Train, Validation e Test'''
# Train:      64%
# Validation: 16%
# Test:       20%

# beginValidateIndex = int(dataset_length * 64 / 100)
# beginTestIndex = int(dataset_length * 80 / 100)

if mode_final_test:
    # carico il test set
    # X_test = X[beginTestIndex:]
    # y_test = y[beginTestIndex:]
    X_test, y_test = read_particles_hdf5(data_directory, num_examples, must_rebalance=must_rebalance, rebalance_seed=rebalance_seed, set='test')
else:
    # carico il validation set
    # X_test = X[beginValidateIndex:beginTestIndex]
    # y_test = y[beginValidateIndex:beginTestIndex]
    X_test, y_test = read_particles_hdf5(data_directory, num_examples, must_rebalance=must_rebalance, rebalance_seed=rebalance_seed, set='validation')

''' Valutazione del modello '''
if model_type == 'conv':
    evaluation = model.evaluate(X_test, y_test)
    print(evaluation)
elif model_type == 'perceptron' or model_type == 'svm':
    X_test = X_test.reshape(len(X_test), -1)

y_pred = model.predict(X_test)

if mode_final_test:
    str_mode = "test"
else:
    str_mode = "validation"

plt.figure(1)
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred)
area_under_curve = auc(recall, precision)
ap = average_precision_score(y_test, y_pred)
plt.title("PR curve for model: " + model_name)
plt.plot(recall, precision, label='AUC = {:.3f}, AP = {:.3f}'.format(area_under_curve, ap))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.savefig(model_directory + "/pr_" + str_mode + ".png")

plt.figure(2)
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred)
area_under_curve = auc(fpr, tpr)
ras = roc_auc_score(y_test, y_pred)
plt.title("ROC curve for model: " + model_name)
plt.plot(fpr, tpr, label='AUC = {:.3f}, RAS = {:.3f}'.format(area_under_curve, ras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='best')
plt.savefig(model_directory + "/roc_" + str_mode + ".png")

print("test concluso, risultati riportati in: " + model_directory)
