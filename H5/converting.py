import h5py
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Convert fully_contained from bool to int64 in HDF5 file")
parser.add_argument("data_path", type=str, help="Path to the input HDF5 file")
args = parser.parse_args()
input_file = args.input_file
output_file = os.path.join(os.path.dirname(input_file), "FC_" + os.path.basename(input_file))

# Open the original file in read mode and the new file in write mode
with h5py.File(input_file, "r") as fin, h5py.File(output_file, "w") as fout:
  for key in fin.keys():
    data = fin[key][:]
    
    # Convert 'fully_contained' from bool to int64
    if key == "fully_contained":
      data = data.astype(np.int64)
    
    # Copy dataset to the new file
    fout.create_dataset(key, data=data)

print("Conversion completed. New file saved as:", output_file)