import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Check contents of an HDF5 file")
parser.add_argument("data_path", type=str, help="Path to the HDF5 file")
args = parser.parse_args()

data_path = args.data_path
h5_file = h5py.File(data_path, "r")
for key, array in h5_file.items():
  print(f"{key:16s} array has shape {array.shape}")
  
event_labels = np.array(h5_file['labels'])
label_set, label_counts = np.unique(event_labels, return_counts=True)
for label, count in zip(label_set, label_counts):
  print(f"There are {count} events with label {label}")
  
fully_contained = h5_file['fully_contained']
print("fully_contained dtype:", fully_contained.dtype)
print("first 10 entries:", fully_contained[:10])