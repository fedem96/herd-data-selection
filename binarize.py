import sys
import numpy as np
import h5py


from utils import read_particles_hdf5, memory_usage

if len(sys.argv) <= 3:
    print("missing parameters")
    exit(1)

particles = sys.argv[1]
dataset_name = sys.argv[2]
threshold = float(sys.argv[3])

dataset_file = "../infn/h5converted/" + particles + "_sphere/data/" + dataset_name

X, y = read_particles_hdf5(dataset_file)

print(X.shape)
print(memory_usage())

X = (X >= threshold).astype(bool)

print(X.shape)
print(memory_usage())

new_dataset_name = dataset_name[:dataset_name.index(".")] + "(b" + str(threshold) + ").hdf5"
new_dataset_file = "../infn/h5converted/" + particles + "_sphere/data/" + new_dataset_name
f = h5py.File(new_dataset_file, "w")
f.create_dataset("examples", data=X, compression='gzip', dtype=bool)
f.create_dataset("labels", data=y, compression='gzip')
f.close()
