""" script che permette di convertire un insieme di file .npz in un unico file .hdf5 """

import os, sys

import h5py
from utils import read_particles

if len(sys.argv) > 2:
    particles = sys.argv[1]
    seed = int(sys.argv[2])
else:
    # tipo di particelle
    particles = "electrons"
    # seed
    seed = 0

# numero di file da leggere, numero <= 0 per leggerli tutti
num_files = 0

# percorso della cartella in cui leggere i dati
data_directory = "../infn/pyconverted/" + particles + "_sphere/data/*.npz"

output_directory = "../infn/h5converted/" + particles + "_sphere/data"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if num_files <= 0:
    str_files = ""
else:
    str_files = "-" + str(num_files) + "_files"

output_file = output_directory + "/dataset" + str_files + "-seed_" + str(seed) + ".hdf5"

X, y = read_particles(data_directory, num_files=num_files, SEED=seed)

f = h5py.File(output_file, "w")
f.create_dataset("examples", data=X, compression='gzip')
f.create_dataset("labels", data=y, compression='gzip')
f.close()
