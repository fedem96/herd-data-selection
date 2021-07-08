import sys
import numpy as np
import h5py
import skimage.measure

from utils import read_particles_hdf5

if len(sys.argv) <= 2:
    print("missing parameters")
    exit(1)

particles = sys.argv[1]
dataset_name = sys.argv[2]

dataset_file = "../infn/h5converted/" + particles + "_sphere/data/" + dataset_name

X, y = read_particles_hdf5(dataset_file)

print(X.shape)

X = skimage.measure.block_reduce(X, (1,2,2,2,1), np.average)

print(X.shape)

new_dataset_file = "../infn/h5converted/" + particles + "_sphere/data/" + dataset_name[:dataset_name.index(".")] + "(a).hdf5"
f = h5py.File(new_dataset_file, "w")
f.create_dataset("examples", data=X, compression='gzip')
f.create_dataset("labels", data=y, compression='gzip')
f.close()
