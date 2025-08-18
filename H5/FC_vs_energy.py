import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Plot FC statistics for train/validation/test splits")
parser.add_argument("data_path", type=str, help="Path to the HDF5 file containing energies and FC labels")
parser.add_argument("split_path", type=str, help="Path to the .npz file containing split indices")
args = parser.parse_args()

split = np.load(args.split_path)
train_idxs = split['train_idxs']
val_idxs = split['val_idxs']
test_idxs = split['test_idxs']

with h5py.File(args.data_path, "r") as f:
  energies = f['energies'][:]  # shape: (N,)
  labels = f['fully_contained'][:]      # Assume FC label, 0 or 1

output_dir = os.path.dirname(args.split_path) + "/FC_stats"

import os
os.makedirs(output_dir, exist_ok=True)

# Define energy bins
energy_bins = np.linspace(0, np.max(energies), 30)

def plot_split(split_name, idxs):
  # ax1 is the main axis for histograms (event counts, right y-axis)
  # ax2 is the secondary axis for FC ratio (line, left y-axis)
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  E = energies[idxs]
  L = labels[idxs]

  total_counts, bins = np.histogram(E, bins=energy_bins)
  FC_counts, _ = np.histogram(E[L == 1], bins=energy_bins)

  bin_centers = 0.5 * (bins[1:] + bins[:-1])
  valid = total_counts > 0
  bin_centers = bin_centers[valid]
  total_counts = total_counts[valid]
  FC_counts = FC_counts[valid]
  ratio = FC_counts / total_counts

  # ax1 handles histograms (event count) -- right y-axis
  ax1.bar(bin_centers, total_counts, width=np.diff(energy_bins)[valid], alpha=0.5, label='Total', color='skyblue', zorder=1)
  ax1.bar(bin_centers, FC_counts, width=np.diff(energy_bins)[valid], alpha=0.5, label='FC', color='sandybrown', zorder=1)
  ax1.set_xlabel('Energy (MeV)')
  ax1.set_ylabel('Event Count')
  ax1.set_xlim(left=min(bin_centers), right=max(bin_centers))
  ax1.set_ylim(bottom=0, top=1.1 * max(total_counts))
  ax1.legend(loc='upper right')
  ax1.yaxis.set_label_position("right")
  ax1.yaxis.tick_right()

  # ax2 handles FC ratio line, color 'tab:blue' -- left y-axis
  ax2.plot(bin_centers, ratio, 'o-', color='tab:blue', label='FC Ratio', zorder=10)
  ax2.set_ylabel('FC Ratio')
  ax2.set_ylim(0, 1.2)
  ax2.set_yticks(np.linspace(0, 1.0, 6))
  ax2.set_title(f'{split_name.capitalize()} Set')
  ax2.legend(loc='upper left')
  ax2.yaxis.set_label_position("left")
  ax2.yaxis.tick_left()
  ax2.set_zorder(ax1.get_zorder() + 1)
  ax2.patch.set_visible(False)

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, f"{split_name}_fc_stats_energy.png"))

# Plot figures
plot_split('train', train_idxs)
plot_split('validation', val_idxs)
plot_split('test', test_idxs)