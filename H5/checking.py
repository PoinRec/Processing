import h5py
import numpy as np

data_path = "/home/zhihao/Data/WCTE_data_fixed/wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"
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