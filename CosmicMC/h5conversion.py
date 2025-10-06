import numpy as np
import argparse
import h5py
import shutil

parser = argparse.ArgumentParser(description="convert cosmic MC HDF5 file to a format compatible with WatChMaL")
parser.add_argument("input_file", type=str, help="Path to the cosmic MC input HDF5 file")
args = parser.parse_args()

output_file = args.input_file.replace('.h5', '_wm.h5')

shutil.copy(args.input_file, output_file)

with h5py.File(output_file, "r+") as f_new:
  
  PMT_Q = np.array(f_new["pmtQ"][...])   # shape (N, 1843)
  PMT_T = np.array(f_new["pmtT"][...])   # shape (N, 1843)
  
  event_hits_index = np.zeros(PMT_Q.shape[0], dtype=np.int32)  # N+1

  hit_charge = []
  hit_time = []
  hit_pmt = []

  for i in range(PMT_Q.shape[0]):
    hits = PMT_Q[i] > 0
    idxs = np.nonzero(hits)[0]
    
    hit_charge.extend(PMT_Q[i, hits].tolist())
    hit_time.extend(PMT_T[i, hits].tolist())
    hit_pmt.extend(idxs.tolist())

    if i != PMT_Q.shape[0] - 1:
      event_hits_index[i + 1] = event_hits_index[i] + len(idxs)
  
  f_new.create_dataset("event_hits_index", data=event_hits_index, dtype=np.int32)
  f_new.create_dataset("hit_charge", data=np.array(hit_charge, dtype=np.float32), dtype=np.float32)
  f_new.create_dataset("hit_time", data=np.array(hit_time, dtype=np.float32), dtype=np.float32)
  f_new.create_dataset("hit_pmt", data=np.array(hit_pmt, dtype=np.int32), dtype=np.int32)
  f_new.create_dataset("event_ids", data=np.arange(PMT_Q.shape[0], dtype=np.int32), dtype=np.int32)

print(f"New file created: {output_file}")