import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

sys.path.append('/home/zhihao/WatChMaL')
import watchmal.utils.math as math


parser = argparse.ArgumentParser(description="Check FC events distribution in HDF5 file")
parser.add_argument("data_path", type=str, help="Path to the HDF5 file")
args = parser.parse_args()

output_dir = os.path.dirname(args.data_path) + "/FC_stats"
os.makedirs(output_dir, exist_ok=True)

tank_half_height= 271.4235 / 2
tank_radius= 307.5926 / 2

with h5py.File(args.data_path, "r") as f:
  labels = f['labels'][:].squeeze()
  FC = f['fully_contained'][:].squeeze()
  positions = f['positions'][:].squeeze()
  angles = f['angles'][:].squeeze()
  energies = f['energies'][:].squeeze()
  
label_set = np.unique(labels)
label_map = {0: "gamma", 1: "e", 2: "mu", 3: "pi"}
towalls = math.towall(positions, angles, tank_half_height=tank_half_height, tank_radius=tank_radius)

for label in label_set:
  mask = FC & (labels == label)

  x = towalls[mask]
  y = energies[mask]

  initial_count = len(x)

  valid = np.isfinite(x) & np.isfinite(y)
  x = x[valid]
  y = y[valid]
  
  final_count = len(x)
  print(f"Label: {label_map[label]}, Initial count: {initial_count}, Valid count: {final_count}")

  if len(x) == 0:
    print(f"No valid events for {label_map[label]}")
    continue

  plt.figure(figsize=(10, 6))
  plt.hist2d(x, y, bins=(100, 100))
  plt.colorbar(label="counts per bin")
  plt.xlabel("towall (cm)")
  plt.ylabel("energy (MeV)")
  plt.title(f"2D histogram of (towall, energy) for FC events ({label_map[label]})")
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, f"{label_map[label]}_energy_distribution.png"))
  plt.close()
  print(f"Saved 2D histogram for {label_map[label]}")

  
  
  
  