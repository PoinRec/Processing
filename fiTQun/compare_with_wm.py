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

output_dir = os.path.join(fq_dir, "comparison")
os.makedirs(output_dir, exist_ok=True)

with h5py.File(args.fitqun_file, 'r') as f_fitqun, h5py.File(args.watchmal_file, 'r') as f_watchmal:
  fq_dir = np.array(f_fitqun['best_dir'][:])
  fq_hypo = np.array(f_fitqun['best_hypo'][:])
  
  fq_mu_minus_e = np.array(f_fitqun['dnll_mu_minus_e'][:])
  # sigmoid to convert to probability
  fq_prob = 1 / (1 + np.exp(fq_mu_minus_e))
  
  fq_mom = np.array(f_fitqun['best_mom'][:])
  fq_energy = math.energy_from_momentum(fq_mom, fq_hypo)
  
  
  fq_nll = np.array(f_fitqun['best_nll'][:])
  fq_pos = np.array(f_fitqun['best_pos'][:])
  fq_seed = np.array(f_fitqun['best_seed'][:])
  fq_nevt = np.array(f_fitqun['nevt'][:])
  fq_valid = np.array(f_fitqun['valid'][:])
  fq_exit_pos = np.array(f_fitqun['exit_points'][:])
  
  wm_FC_mu = np.array(f_watchmal['FC_mu_softmax'][:])
  
  wm_softmax = np.array(f_watchmal['classification_softmax'][:])
  wm_prob = wm_softmax[:,1] / (wm_softmax[:,0] + wm_softmax[:,1])
  
  wm_dir = np.array(f_watchmal['direction_mu_predicted_directions'][:])
  wm_energy = np.array(f_watchmal['energy_mu_predicted_energies'][:]).squeeze()
  wm_pos = np.array(f_watchmal['position_mu_predicted_positions'][:])
  wm_exit_pos = np.array(f_watchmal['exit_points'][:])

print(np.unique(fq_hypo, return_counts=True))

energy_diff = fq_energy - wm_energy
plt.hist(energy_diff, bins=50, range=(energy_diff.min(), energy_diff.max()), histtype='step', color='b', linewidth=1.5, label="Energy")

plt.xlabel("Energy difference[MeV]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "energy_difference.png"))
plt.close()


dir_diff = math.angle_between_directions(fq_dir, wm_dir, degrees=True)
plt.hist(dir_diff, bins=50, range=(dir_diff.min(), dir_diff.max()), histtype='step', color='b', linewidth=1.5, label="Direction")

plt.xlabel("Direction difference[°]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "direction_difference.png"))
plt.close()

pos_diff = np.linalg.norm(fq_pos - wm_pos, axis=-1)
plt.hist(pos_diff, bins=50, range=(pos_diff.min(), pos_diff.max()), histtype='step', color='b', linewidth=1.5, label="Position")

plt.xlabel("Position difference[cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "position_difference.png"))
plt.close()

exit_pos_diff = np.linalg.norm(fq_exit_pos - wm_exit_pos, axis=-1)
plt.hist(exit_pos_diff, bins=50, range=(exit_pos_diff.min(), exit_pos_diff.max()), histtype='step', color='b', linewidth=1.5, label="Exit Position")

plt.xlabel("Exiting Position difference[cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "exit_position_difference.png"))
plt.close()


# Normalize the classification metrics for comparison

prob_diff = fq_prob - wm_prob
plt.hist(prob_diff, bins=50, range=(prob_diff.min(), prob_diff.max()), histtype='step', linewidth=1.5, label="Probability")

plt.xlabel("Classification difference")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "classification_compare.png"))