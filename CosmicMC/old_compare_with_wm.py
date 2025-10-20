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
parser.add_argument("selection_fq_h5_file", type=str, help="Path to the fiTQun selection HDF5 file")
args = parser.parse_args()

fq_dir = os.path.dirname(args.fitqun_file)

output_dir = os.path.join(fq_dir, "Comparison")
os.makedirs(output_dir, exist_ok=True)


with h5py.File(args.fitqun_file, 'r') as f_fitqun, h5py.File(args.watchmal_file, 'r') as f_watchmal, h5py.File(args.selection_fq_h5_file, 'r') as f_selection:
  
  pass_selection = np.array(f_selection['pass_selection'])
  sorted_indices = np.array(f_watchmal['sorted_indices'])
  
  indices_mask = np.zeros_like(pass_selection, dtype=bool)
  indices_mask[sorted_indices] = True
  
  fq_selection = pass_selection & indices_mask
  wm_selection = pass_selection[sorted_indices]
  
  print("Number of selected events:", np.sum(fq_selection))
  print("Number of selected events in WatChMaL:", np.sum(wm_selection))
  
  
  fq_dir = np.array(f_fitqun['best_dir'])[fq_selection]
  fq_hypo = np.array(f_fitqun['best_hypo'])[fq_selection]
  
  fq_mu_minus_e = np.array(f_fitqun['dnll_mu_minus_e'])[fq_selection]
  # sigmoid to convert to probability
  fq_prob = 1 / (1 + np.exp(fq_mu_minus_e))
  
  fq_mom = np.array(f_fitqun['best_mom'])[fq_selection]
  fq_energy = math.energy_from_momentum(fq_mom, fq_hypo)
  
  
  fq_nll = np.array(f_fitqun['best_nll'])[fq_selection]
  fq_pos = np.array(f_fitqun['best_pos'])[fq_selection]
  fq_seed = np.array(f_fitqun['best_seed'])[fq_selection]
  fq_nevt = np.array(f_fitqun['nevt'])[fq_selection]
  fq_valid = np.array(f_fitqun['valid'])[fq_selection]
  fq_exit_pos = np.array(f_fitqun['exit_points'])[fq_selection]
  
  
  
  wm_FC_mu = np.array(f_watchmal['FC_mu_softmax'][:])[wm_selection]
  
  wm_softmax = np.array(f_watchmal['classification_softmax'][:])[wm_selection]
  wm_prob = wm_softmax[:,1] / (wm_softmax[:,0] + wm_softmax[:,1])
  
  wm_dir = np.array(f_watchmal['direction_mu_predicted_directions'][:])[wm_selection]
  
  wm_dir = wm_dir / np.linalg.norm(wm_dir, axis=-1, keepdims=True)
  
  wm_energy = np.array(f_watchmal['energy_mu_predicted_energies'][:]).squeeze()[wm_selection]
  wm_pos = np.array(f_watchmal['position_mu_predicted_positions'][:])[wm_selection]
  wm_exit_pos = np.array(f_watchmal['exit_points'][:])[wm_selection]

print(np.unique(fq_hypo, return_counts=True))

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





valid_a = np.isfinite(fq_energy)
valid_b = np.isfinite(wm_energy)
lo = np.nanmin([np.nanmin(fq_energy[valid_a]) if valid_a.any() else np.nan, np.nanmin(wm_energy[valid_b]) if valid_b.any() else np.nan])
hi = np.nanmax([np.nanmax(fq_energy[valid_a]) if valid_a.any() else np.nan, np.nanmax(wm_energy[valid_b]) if valid_b.any() else np.nan])
if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
  lo, hi = 0.0, 1.0
plt.figure()
if valid_a.any():
  plt.hist(fq_energy[valid_a], bins=60, range=(0, 2000), histtype='step', linewidth=1.5, label="FQ Energy")
if valid_b.any():
  plt.hist(wm_energy[valid_b], bins=60, range=(0, 2000), histtype='step', linewidth=1.5, label="WM Energy")
plt.xlabel("Energy [MeV]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ZZ_overall_energy_fq_vs_wm.png"))
plt.close()

valid_a = np.isfinite(fq_prob)
valid_b = np.isfinite(wm_prob)
plt.figure()
if valid_a.any():
  plt.hist(fq_prob[valid_a], bins=60, range=(0.0, 1.0), histtype='step', linewidth=1.5, label="FQ P(mu)")
if valid_b.any():
  plt.hist(wm_prob[valid_b], bins=60, range=(0.0, 1.0), histtype='step', linewidth=1.5, label="WM P(mu)")
plt.xlabel("Classification probability")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ZZ_overall_classprob_fq_vs_wm.png"))
plt.close()

fq_pos_y = np.asarray(fq_pos)[:, 1]
fq_exit_y = np.asarray(fq_exit_pos)[:, 1]
valid2d = np.isfinite(fq_pos_y) & np.isfinite(fq_exit_y)
if np.any(valid2d):
  x = fq_pos_y[valid2d]
  y = fq_exit_y[valid2d]
  xlo, xhi = np.percentile(x, [0.5, 99.5])
  ylo, yhi = np.percentile(y, [0.5, 99.5])
  plt.figure()
  plt.hist2d(x, y, bins=100, range=[[xlo, xhi], [ylo, yhi]], cmap="viridis")
  plt.xlabel("FQ vertex y [cm]")
  plt.ylabel("FQ exit y [cm]")
  plt.colorbar(label="Counts")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "ZZ_fq_exit_y_vs_fq_vertex_y_2D.png"))
  plt.close()

  plt.figure()
  plt.hist(y, bins=60, histtype='step', linewidth=1.5)
  plt.xlabel("FQ exit y [cm]")
  plt.ylabel("Counts")
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "ZZ_fq_exit_y_1D.png"))
  plt.close()