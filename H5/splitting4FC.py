import h5py
import numpy as np
import sys
import argparse
import os

sys.path.append('/home/zhihao/WatChMaL')
import watchmal.utils.math as math

parser = argparse.ArgumentParser(description="Split FC dataset into train/val/test for different labels")
parser.add_argument("data_path", type=str, help="Path to the input HDF5 file")
args = parser.parse_args()
data_path = args.data_path

output_dir = os.path.join(os.path.dirname(data_path), "Splitting")
os.makedirs(output_dir, exist_ok=True)
label_map = {0: "gamma", 1: "e", 2: "mu", 3: "pi"}


h5_file = h5py.File(data_path, "r")

event_labels = np.array(h5_file['labels'])
label_set, label_counts = np.unique(event_labels, return_counts=True)
print(f'Label set: {label_set}, Counts: {label_counts}')


idxs_paths = [os.path.join(output_dir, f"split_list_{label_map[label]}_FC.npz") for label in label_set]

nhit_threshold = 10
nhit_test_threshold = 25
# towall_test_threshold = 100

validation_proportion = 0.1
training_proportion = 0.8

tank_half_height= 271.4235 / 2
tank_radius= 307.5926 / 2


# to determine number of hits per event, take the difference between adjacent entries in the event hits index
event_hits_index = np.array(h5_file["event_hits_index"])
nhits = np.diff(event_hits_index, append=h5_file["hit_pmt"].shape[0])

h5_file.close()


# train on fully contained events with more than (nhit_threshold) hits
selection = (nhits > nhit_threshold)
test_selection = (nhits > nhit_test_threshold)

event_indices = np.arange(len(event_labels))

testing_indices = np.array([], dtype=int)
validation_indices = np.array([], dtype=int)
training_indices = np.array([], dtype=int)

validation_proportion_start = 1 - validation_proportion - training_proportion
training_proportion_start = 1 - training_proportion

for label, count, idxs_path in zip(label_set, label_counts, idxs_paths):
  selected_indices = event_indices[selection & (event_labels==label)]
  selected_count = len(selected_indices)
  print(f'label: {label}, selected_count: {selected_count}')
  
  validation_indices = selected_indices[int(validation_proportion_start * selected_count):int(training_proportion_start * selected_count)]
  training_indices = selected_indices[int(training_proportion_start * selected_count):]

  testing_selection = test_selection[selected_indices]
  testing_selection = testing_selection[:int(validation_proportion_start * selected_count)]
  test_indices = selected_indices[:int(validation_proportion_start * selected_count)]
  testing_indices = test_indices[testing_selection]
  print(f'label: {label}, test_selected_count: {len(testing_indices)}')
  
  np.savez(idxs_path,
  test_idxs=testing_indices,
  val_idxs=validation_indices,
  train_idxs=training_indices)