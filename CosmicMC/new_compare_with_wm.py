import numpy as np
import h5py
import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/zhihao/WatChMaL')

import watchmal.utils.math as math

parser = argparse.ArgumentParser(description="Comparison script between fiTQun and WatChMaL classification outputs")
parser.add_argument("cosmic_mc_file", type=str, help="Path to the cosmic MC converted HDF5 file")
parser.add_argument("--watchmal_output", type=str,help="Path to the WatChMaL output HDF5 file. If not provided, it is assumed to be in the same directory as the cosmic_mc_file with name 'Predict/merged_outputs.h5'")

args = parser.parse_args()

if args.watchmal_output is None:
  args.watchmal_output = os.path.join(os.path.dirname(args.cosmic_mc_file), "Predict", "merged_outputs.h5")

print("Cosmic file:", args.cosmic_mc_file)
print("WatChMaL output:", args.watchmal_output)



fq_vs_wm = os.path.join(os.path.dirname(args.cosmic_mc_file), "Comparison/fq_vs_wm")
os.makedirs(fq_vs_wm, exist_ok=True)

wm_vs_true = os.path.join(os.path.dirname(args.cosmic_mc_file), "Comparison/recon_vs_true")
os.makedirs(wm_vs_true, exist_ok=True)


with (h5py.File(args.cosmic_mc_file, 'r') as fq_f, h5py.File(args.watchmal_output, 'r') as wm_f):
  
  pass_selection = np.array(fq_f['pass_selection'])
  sorted_indices = np.array(wm_f['sorted_indices'])
  
  indices_mask = np.zeros_like(pass_selection, dtype=bool)
  indices_mask[sorted_indices] = True
  
  fq_selection = pass_selection & indices_mask
  wm_selection = pass_selection[sorted_indices]
  
  print("Number of selected events:", np.sum(fq_selection))
  print("Number of selected events in WatChMaL:", np.sum(wm_selection))
  
  true_dir = np.array(fq_f['direction'])[fq_selection]
  true_entrance_pos = np.array(fq_f['entrance_pos'])[fq_selection] / 10.0  # convert to cm
  true_exit_pos = np.array(fq_f['exit_pos'])[fq_selection] / 10.0  # convert to cm
  true_mom = np.array(fq_f['momentum']).squeeze()[fq_selection] 
  true_energy = math.energy_from_momentum(true_mom, 2)   # cannot compared with energies reconstructed because these are energies at vertex
  true_vertex = np.array(fq_f['vertex'])[fq_selection] / 10.0  # convert to cm
  
  
  fq_dir = np.array(fq_f['fq_direction'])[fq_selection]
  fq_mom = np.array(fq_f['fq_momentum']).squeeze()[fq_selection]
  fq_energy = math.energy_from_momentum(fq_mom, 2)
  fq_entrance_pos = np.array(fq_f['fq_entrance_pos'])[fq_selection] / 10.0  # convert to cm
  fq_exit_pos = np.array(fq_f['fq_exit_pos'])[fq_selection] / 10.0  # convert to cm
  
  wm_dir = np.array(wm_f['direction_mu_predicted_directions'])[wm_selection][:, [0, 2, 1]]
  wm_dir[:, 1] *= -1
  
  ''' Normalize direction vectors (very important !!!) '''
  wm_dir = wm_dir / np.linalg.norm(wm_dir, axis=-1, keepdims=True)
  
  wm_energy = np.array(wm_f['energy_mu_predicted_energies']).squeeze()[wm_selection]
  
  wm_pos = np.array(wm_f['position_mu_predicted_positions'])[wm_selection][:, [0, 2, 1]]
  wm_pos[:, 1] *= -1
  
  wm_exit_pos = np.array(wm_f['exit_points'])[wm_selection][:, [0, 2, 1]]
  wm_exit_pos[:, 1] *= -1
  



energy_diff = fq_energy - wm_energy
plt.hist(energy_diff, bins=100, range=(energy_diff.min(), energy_diff.max()), histtype='step', color='b', linewidth=1.5, label="Energy")

plt.xlabel("Energy difference[MeV]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fq_vs_wm, "energy_difference.png"))
plt.close()


dir_diff = math.angle_between_directions(fq_dir, wm_dir, degrees=True)
plt.hist(dir_diff, bins=100, range=(dir_diff.min(), dir_diff.max()), histtype='step', color='b', linewidth=1.5, label="Direction")

plt.xlabel("Direction difference[°]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fq_vs_wm, "direction_difference.png"))
plt.close()

pos_diff = np.linalg.norm(fq_entrance_pos - wm_pos, axis=-1)
plt.hist(pos_diff, bins=100, range=(pos_diff.min(), pos_diff.max()), histtype='step', color='b', linewidth=1.5, label="Position")

plt.xlabel("Entrance Position difference[cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fq_vs_wm, "entrance_position_difference.png"))
plt.close()


exit_pos_diff = np.linalg.norm(fq_exit_pos - wm_exit_pos, axis=-1)
plt.hist(exit_pos_diff, bins=100, range=(exit_pos_diff.min(), exit_pos_diff.max()), histtype='step', color='b', linewidth=1.5, label="Exit Position")

plt.xlabel("Exiting Position difference[cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fq_vs_wm, "exit_position_difference.png"))
plt.close()

'''
a = true_energy[np.isfinite(true_energy)]
b = fq_energy[np.isfinite(fq_energy)]
c = wm_energy[np.isfinite(wm_energy)]
lo = np.nanmin([a.min() if a.size else np.nan, b.min() if b.size else np.nan, c.min() if c.size else np.nan])
hi = np.nanmax([a.max() if a.size else np.nan, b.max() if b.size else np.nan, c.max() if c.size else np.nan])
if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
  lo, hi = 0.0, 1.0
plt.figure()
if a.size:
  plt.hist(a, bins=60, range=(lo, hi), histtype='step', linewidth=1.5, label="True Energy")
if b.size:
  plt.hist(b, bins=60, range=(lo, hi), histtype='step', linewidth=1.5, label="FQ Energy")
if c.size:
  plt.hist(c, bins=60, range=(lo, hi), histtype='step', linewidth=1.5, label="WM Energy")
plt.xlabel("Energy [MeV]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(fq_vs_wm, "overall_energy_true_fq_wm.png"))
plt.close()
'''

'''Recon vs True plots'''

# ---- Energy ----
energy_diff_fq = fq_energy - true_energy
energy_diff_wm = wm_energy - true_energy
all_energy_diff = np.concatenate([energy_diff_fq, energy_diff_wm])
bins = np.linspace(-7000, -3000, 100)

plt.hist(energy_diff_fq, bins=bins, histtype='step', color='r', linewidth=1.5, label="fiTQun - True")
plt.hist(energy_diff_wm, bins=bins, histtype='step', color='b', linewidth=1.5, label="WatChMaL - True")
plt.xlabel("Energy difference [MeV]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(wm_vs_true, "energy_difference_recon_vs_true.png"))
plt.close()

print("Mean energy difference (fiTQun - True):", np.mean(energy_diff_fq))
print("Mean energy difference (WatChMaL - True):", np.mean(energy_diff_wm))


# ---- Direction ----
dir_diff_fq = math.angle_between_directions(fq_dir, true_dir, degrees=True)
dir_diff_wm = math.angle_between_directions(wm_dir, true_dir, degrees=True)
all_dir_diff = np.concatenate([dir_diff_fq, dir_diff_wm])
bins = np.linspace(0, 30, 100)

plt.hist(dir_diff_fq, bins=bins, histtype='step', color='r', linewidth=1.5, label="fiTQun vs True")
plt.hist(dir_diff_wm, bins=bins, histtype='step', color='b', linewidth=1.5, label="WatChMaL vs True")
plt.xlabel("Direction difference [°]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(wm_vs_true, "direction_difference_recon_vs_true.png"))
plt.close()

print("Mean direction difference (fiTQun vs True):", np.mean(dir_diff_fq))
print("Mean direction difference (WatChMaL vs True):", np.mean(dir_diff_wm))

# ---- Position ----
pos_diff_fq = np.linalg.norm(fq_entrance_pos - true_entrance_pos, axis=-1)
pos_diff_wm = np.linalg.norm(wm_pos - true_entrance_pos, axis=-1)
all_pos_diff = np.concatenate([pos_diff_fq, pos_diff_wm])
bins = np.linspace(0, 100, 100)

plt.hist(pos_diff_fq, bins=bins, histtype='step', color='r', linewidth=1.5, label="fiTQun vs True")
plt.hist(pos_diff_wm, bins=bins, histtype='step', color='b', linewidth=1.5, label="WatChMaL vs True")
plt.xlabel("Entrance Position difference [cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(wm_vs_true, "entrance_position_difference_recon_vs_true.png"))
plt.close()

print("Mean entrance position difference (fiTQun vs True):", np.mean(pos_diff_fq))
print("Mean entrance position difference (WatChMaL vs True):", np.mean(pos_diff_wm))

# ---- Exit Position ----
exit_pos_diff_fq = np.linalg.norm(fq_exit_pos - true_exit_pos, axis=-1)
exit_pos_diff_wm = np.linalg.norm(wm_exit_pos - true_exit_pos, axis=-1)
all_exit_pos_diff = np.concatenate([exit_pos_diff_fq, exit_pos_diff_wm])
bins = np.linspace(0, 100, 100)

plt.hist(exit_pos_diff_fq, bins=bins, histtype='step', color='r', linewidth=1.5, label="fiTQun vs True")
plt.hist(exit_pos_diff_wm, bins=bins, histtype='step', color='b', linewidth=1.5, label="WatChMaL vs True")
plt.xlabel("Exiting Position difference [cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(wm_vs_true, "exit_position_difference_recon_vs_true.png"))
plt.close()

print("Mean exiting position difference (fiTQun vs True):", np.mean(exit_pos_diff_fq))
print("Mean exiting position difference (WatChMaL vs True):", np.mean(exit_pos_diff_wm))

# ---- D_MIN ----
delta_fq = true_vertex - fq_entrance_pos
delta_wm = true_vertex - wm_pos

d_min_fq = np.sqrt(np.linalg.norm(delta_fq, axis=-1) ** 2 - np.einsum('ij,ij->i', delta_fq, true_dir) ** 2)
d_min_wm = np.sqrt(np.linalg.norm(delta_wm, axis=-1) ** 2 - np.einsum('ij,ij->i', delta_wm, true_dir) ** 2)

bins = np.linspace(0, 100, 100)

plt.hist(d_min_fq, bins=bins, histtype='step', color='r', linewidth=1.5, label="fiTQun D_min")
plt.hist(d_min_wm, bins=bins, histtype='step', color='b', linewidth=1.5, label="WatChMaL D_min")
plt.xlabel("D_min [cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(wm_vs_true, "d_min_recon_vs_true.png"))
plt.close() 


# ---- Z-Axis Position ----
entrance_z_fq = fq_entrance_pos[:, 2]
entrance_z_wm = wm_pos[:, 2]
true_z = true_entrance_pos[:, 2]
bins = np.linspace(110, 140, 100)

plt.hist(entrance_z_fq, bins=bins, histtype='step', color='r', linewidth=1.5, label="fiTQun Entrance Z")
plt.hist(entrance_z_wm, bins=bins, histtype='step', color='b', linewidth=1.5, label="WatChMaL Entrance Z")
plt.hist(true_z, bins=bins, histtype='step', color='g', linewidth=1.5, label="True Entrance Z")
plt.xlabel("Entrance Z Position [cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(wm_vs_true, "entrance_z_position_recon_vs_true.png"))
plt.close()