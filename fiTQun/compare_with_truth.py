import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import h5py

import sys
sys.path.append('/home/zhihao/WatChMaL')
import watchmal.utils.math as math

parser = argparse.ArgumentParser(description="Comparison script between fiTQun outputs and truth information for the cosmic MC dataset")
parser.add_argument("fitqun_file_h5_file", type=str, help="Path to the fiTQun HDF5 file")
args = parser.parse_args()

fq_dir = os.path.dirname(args.fitqun_file_h5_file)
output_dir = os.path.join(fq_dir, "Truth_Comparison_fq")

os.makedirs(output_dir, exist_ok=True)

with h5py.File(args.fitqun_file_h5_file, 'r') as f_fitqun:
  fq_direction = np.array(f_fitqun['fq_direction'][:])
  fq_entrance_pos = np.array(f_fitqun['fq_entrance_pos'][:])
  fq_exit_pos = np.array(f_fitqun['fq_exit_pos'][:])
  fq_momentum = np.array(f_fitqun['fq_momentum'][:])      # fiTQun reconstructed momentum

  direction = np.array(f_fitqun['direction'][:])
  entrance_pos = np.array(f_fitqun['entrance_pos'][:])
  exit_pos = np.array(f_fitqun['exit_pos'][:])
  momentum = np.array(f_fitqun['momentum'][:])            # true momentum at vertex, usually much 
  
  is_top_down = np.array(f_fitqun['is_top_down'][:], dtype=bool)
  pass_selection = np.array(f_fitqun['pass_selection'][:], dtype=bool)

  n_events = fq_direction.shape[0]
  
print(is_top_down.sum(), "top-down events out of", n_events)
print(pass_selection.sum(), "events pass selection out of", n_events)

print(fq_momentum)
print(momentum)

direction_diff = math.angle_between_directions(fq_direction, direction, degrees=True)
plt.hist(direction_diff[pass_selection], bins=50, range=(0, direction_diff.max()), histtype='step', color='b', linewidth=1.5, label="Direction")
plt.xlabel("Direction difference [degrees]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "direction_difference.png"))
plt.close()

entrance_pos_diff = np.linalg.norm(fq_entrance_pos - entrance_pos, axis=1)
plt.hist(entrance_pos_diff[pass_selection], bins=50, range=(0, 750), histtype='step', color='b', linewidth=1.5, label="Entrance Position")
plt.xlabel("Entrance Position difference [cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "entrance_position_difference.png"))
plt.close()

exit_pos_diff = np.linalg.norm(fq_exit_pos - exit_pos, axis=1)
plt.hist(exit_pos_diff[pass_selection], bins=50, range=(0, 750), histtype='step', color='b', linewidth=1.5, label="Exit Position")
plt.xlabel("Exit Position difference [cm]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "exit_position_difference.png"))
plt.close()

'''
momentum_diff = fq_momentum - momentum
plt.hist(momentum_diff[pass_selection], bins=50, range=(-20000, 0), histtype='step', color='b', linewidth=1.5, label="Momentum")
plt.xlabel("Momentum difference [MeV/c]")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "momentum_difference.png"))
plt.close()
'''