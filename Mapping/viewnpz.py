import numpy as np
import sys

def view_npz(filename, num_elements=10):
  data = np.load(filename)
  print(f"File '{filename}' contains the following arrays:")
  for name in data.files:
    arr = data[name]
    print(f"\nArray name: {name}")
    print(f"Shape: {arr.shape}")
    print(f"First {num_elements} elements: {arr.flatten()[:num_elements]}")

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python3 /home/zhihao/Processing/Mapping/viewnpz.py /home/zhihao/Data/WCTE_data_fixed/geofile_wcte.npz")
  else:
    view_npz(sys.argv[1])