import h5py
import numpy as np

# Input and output file paths
input_file = "/home/zhihao/Data/WCTE_data_fixed/wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"
output_file = "FC_wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"

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