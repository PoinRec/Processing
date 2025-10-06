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

wm_vs_true = os.path.join(os.path.dirname(args.cosmic_mc_file), "Comparison/wm_vs_true")
os.makedirs(wm_vs_true, exist_ok=True)

split_idxs = np.load(os.path.join(os.path.dirname(args.cosmic_mc_file), "dummy_split_list.npz"))["test_idxs"]


with (# h5py.File(args.cosmic_mc_file, 'r') as fq_f, 
      h5py.File(args.watchmal_output, 'r') as wm_f
):
  '''
  true_dir = np.array(fq_f['direction'][split_idxs])
  true_entrance_pos = np.array(fq_f['entrance_pos'][split_idxs])
  true_exit_pos = np.array(fq_f['exit_pos'][split_idxs])
  true_mom = np.array(fq_f['momentum'][split_idxs]).squeeze()
  true_energy = math.energy_from_momentum(true_mom, 2)
  true_vertex = np.array(fq_f['vertex'][split_idxs])
  
  
  fq_dir = np.array(fq_f['fq_direction'][split_idxs])
  fq_mom = np.array(fq_f['fq_momentum'][split_idxs]).squeeze()
  fq_energy = math.energy_from_momentum(fq_mom, 2)
  fq_entrance_pos = np.array(fq_f['fq_entrance_pos'][split_idxs])
  fq_exit_pos = np.array(fq_f['fq_exit_pos'][split_idxs])
  '''
  
  wm_dir = np.array(wm_f['direction_mu_predicted_directions'])
  wm_energy = np.array(wm_f['energy_mu_predicted_energies']).squeeze()
  wm_pos = np.array(wm_f['position_mu_predicted_positions'])
  wm_exit_pos = np.array(wm_f['exit_points'])



energy_diff = fq_energy - wm_energy
plt.hist(energy_diff, bins=50, range=(energy_diff.min(), energy_diff.max()), histtype='step', color='b', linewidth=1.5, label="Energy")

plt.xlabel("Energy difference[MeV]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fq_vs_wm, "energy_difference.png"))
plt.close()


dir_diff = math.angle_between_directions(fq_dir, wm_dir, degrees=True)
plt.hist(dir_diff, bins=50, range=(dir_diff.min(), dir_diff.max()), histtype='step', color='b', linewidth=1.5, label="Direction")

plt.xlabel("Direction difference[°]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fq_vs_wm, "direction_difference.png"))
plt.close()

pos_diff = np.linalg.norm(fq_entrance_pos - wm_pos, axis=-1)
plt.hist(pos_diff, bins=50, range=(pos_diff.min(), pos_diff.max()), histtype='step', color='b', linewidth=1.5, label="Position")

plt.xlabel("Position difference[cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fq_vs_wm, "position_difference.png"))
plt.close()

exit_pos_diff = np.linalg.norm(fq_exit_pos - wm_exit_pos, axis=-1)
plt.hist(exit_pos_diff, bins=50, range=(exit_pos_diff.min(), exit_pos_diff.max()), histtype='step', color='b', linewidth=1.5, label="Exit Position")

plt.xlabel("Exiting Position difference[cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fq_vs_wm, "exit_position_difference.png"))
plt.close()