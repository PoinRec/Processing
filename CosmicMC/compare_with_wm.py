import numpy as np
import h5py
import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/zhihao/WatChMaL')

import watchmal.utils.math as math

parser = argparse.ArgumentParser(description="Comparison script between fiTQun and WatChMaL classification outputs")
parser.add_argument("fitqun_file", type=str, help="Path to the fiTQun HDF5 file")
parser.add_argument("watchmal_file", type=str, help="Path to the WatChMaL HDF5 file")
args = parser.parse_args()

fq_dir = os.path.dirname(args.fitqun_file)

output_dir = os.path.join(fq_dir, "Comparison")
os.makedirs(output_dir, exist_ok=True)

split_idxs = np.load(os.path.join(os.path.dirname(args.fitqun_file), "dummy_split_list.npz"))["test_idxs"]

with h5py.File(args.fitqun_file, 'r') as f_fitqun, h5py.File(args.watchmal_file, 'r') as f_watchmal:
  fq_dir = np.array(f_fitqun['best_dir'][split_idxs])
  fq_hypo = np.array(f_fitqun['best_hypo'][split_idxs])
  
  fq_mu_minus_e = np.array(f_fitqun['dnll_mu_minus_e'][split_idxs])
  # sigmoid to convert to probability
  fq_prob = 1 / (1 + np.exp(fq_mu_minus_e))
  
  fq_mom = np.array(f_fitqun['best_mom'][split_idxs])
  fq_energy = math.energy_from_momentum(fq_mom, fq_hypo)
  
  
  fq_nll = np.array(f_fitqun['best_nll'][split_idxs])
  fq_pos = np.array(f_fitqun['best_pos'][split_idxs])
  fq_seed = np.array(f_fitqun['best_seed'][split_idxs])
  fq_nevt = np.array(f_fitqun['nevt'][split_idxs])
  fq_valid = np.array(f_fitqun['valid'][split_idxs])
  fq_exit_pos = np.array(f_fitqun['exit_points'][split_idxs])
  
  wm_FC_mu = np.array(f_watchmal['FC_mu_softmax'][:])
  
  wm_softmax = np.array(f_watchmal['classification_softmax'][:])
  wm_prob = wm_softmax[:,1] / (wm_softmax[:,0] + wm_softmax[:,1])
  
  wm_dir = np.array(f_watchmal['direction_mu_predicted_directions'][:])
  wm_energy = np.array(f_watchmal['energy_mu_predicted_energies'][:]).squeeze()
  wm_pos = np.array(f_watchmal['position_mu_predicted_positions'][:])
  wm_exit_pos = np.array(f_watchmal['exit_points'][:])

print(np.unique(fq_hypo, return_counts=True))

def check_nans(name, arr):
  arr = np.asarray(arr)
  n_total = arr.size
  n_nan = np.count_nonzero(~np.isfinite(arr))
  print(f"{name}: {n_nan}/{n_total} = {n_nan/n_total*100:.3f}% NaN or Inf")

check_nans("fq_energy", fq_energy)
check_nans("wm_energy", wm_energy)

check_nans("fq_dir", fq_dir)
check_nans("wm_dir", wm_dir)

check_nans("fq_pos", fq_pos)
check_nans("wm_pos", wm_pos)

check_nans("fq_exit_pos", fq_exit_pos)
check_nans("wm_exit_pos", wm_exit_pos)

check_nans("fq_prob", fq_prob)
check_nans("wm_prob", wm_prob)

energy_diff = fq_energy - wm_energy
valid = np.isfinite(energy_diff)
plt.hist(energy_diff[valid], bins=50, range=(-600, 200), histtype='step', color='b', linewidth=1.5, label="Energy (Valid)")

plt.xlabel("Energy difference[MeV]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "energy_difference.png"))
plt.close()


dir_diff = math.angle_between_directions(fq_dir, wm_dir, degrees=True)
valid = np.isfinite(dir_diff)
plt.hist(dir_diff[valid], bins=50, range=(dir_diff[valid].min(), dir_diff[valid].max()), histtype='step', color='b', linewidth=1.5, label="Direction")

plt.xlabel("Direction difference[°]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "direction_difference.png"))
plt.close()

pos_diff = np.linalg.norm(fq_pos - wm_pos, axis=-1)
valid = np.isfinite(pos_diff)
plt.hist(pos_diff[valid], bins=50, range=(0, 200), histtype='step', color='b', linewidth=1.5, label="Position")

plt.xlabel("Position difference[cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "position_difference.png"))
plt.close()

exit_pos_diff = np.linalg.norm(fq_exit_pos - wm_exit_pos, axis=-1)
valid = np.isfinite(exit_pos_diff)
plt.hist(exit_pos_diff[valid], bins=50, range=(exit_pos_diff[valid].min(), exit_pos_diff[valid].max()), histtype='step', color='b', linewidth=1.5, label="Exit Position")

plt.xlabel("Exiting Position difference[cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "exit_position_difference.png"))
plt.close()


# Normalize the classification metrics for comparison

prob_diff = fq_prob - wm_prob
valid = np.isfinite(prob_diff)
plt.hist(prob_diff[valid], bins=50, range=(prob_diff[valid].min(), prob_diff[valid].max()), histtype='step', linewidth=1.5, label="Probability")

plt.xlabel("Classification difference")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "classification_compare.png"))