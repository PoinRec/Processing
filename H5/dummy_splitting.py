import h5py
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Create a dummy splitting file")
parser.add_argument("data_path", type=str, help="Path to the input HDF5 file")
args = parser.parse_args()
data_path = args.data_path

output_dir = os.path.dirname(data_path)
idx_path = os.path.join(output_dir, "dummy_split_list.npz")

with h5py.File(data_path, "r") as h5_file:
  event_ids = np.array(h5_file['event_ids'])

print(f'Total number of events: {len(event_ids)}')
print(f'Event IDs range from {event_ids.min()} to {event_ids.max()}')
print(f'Event IDs: {event_ids}')

np.savez(idx_path,
test_idxs=event_ids)

print(f'Dummy splitting file saved to {idx_path}')