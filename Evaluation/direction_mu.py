import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import sys
import argparse

sys.path.append('/home/zhihao/WatChMaL')

data_path = "/home/zhihao/Data/WCTE_data_fixed/wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"
idxs_path = "/home/zhihao/Data/WCTE_data_fixed/Splitting/split_list_mu_regression.npz"

parser = argparse.ArgumentParser(description="Evaluation script")
parser.add_argument("run_dir", type=str, help="Path to muon direction regression run directory")
args = parser.parse_args()
regression_run_dir = args.run_dir
results_dir = regression_run_dir + "/results/"

import os
os.makedirs(results_dir, exist_ok=True)

tank_half_height= 271.4235 / 2
tank_radius= 307.5926 / 2

h5_file = h5py.File(data_path, "r")

import analysis.regression as reg
import analysis.utils.binning as bins
import watchmal.utils.math as math

test_idxs  = np.load(idxs_path)['test_idxs']
test_event_labels = np.array(h5_file['labels'])[test_idxs].squeeze()
test_event_energies = np.array(h5_file['energies'])[test_idxs].squeeze()
test_event_angles = np.array(h5_file['angles'])[test_idxs].squeeze()
test_event_positions = np.array(h5_file['positions'])[test_idxs].squeeze()



test_event_towall = math.towall(test_event_positions, test_event_angles, tank_half_height=tank_half_height, tank_radius=tank_radius)
test_event_directions = math.direction_from_angles(test_event_angles)

# Muon events
direction_regression_output_mu = reg.WatChMaLDirectionRegression(regression_run_dir, "Muon Regression", test_event_directions, test_idxs,)


fig, ax = direction_regression_output_mu.plot_training_progression()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(results_dir + "Loss.png", bbox_inches='tight')


with open(results_dir + "resolution.txt", "w") as f:
  direction_resolution = np.quantile(direction_regression_output_mu.direction_errors, 0.68)
  print(f"Overall direction resolution (68th percentile of direction errors) = {direction_resolution} °")
  f.write(f"Overall direction resolution (68th percentile of direction errors) = {direction_resolution} °")





E_min_val = test_event_energies.min()
E_max_val = test_event_energies.max()
E_binning = bins.get_binning(test_event_energies, 20, E_min_val, E_max_val)


towall_min_val = test_event_towall.min()
towall_max_val = test_event_towall.max()
towall_binning = bins.get_binning(test_event_towall, 20, towall_min_val, towall_max_val)


fig, ax = reg.plot_resolution_profile([direction_regression_output_mu], 'direction_errors', E_binning, x_label="True particle energy [MeV]", y_label="Direction resolution [°]", y_lim=(0,15), errors=True, x_errors=False)
plt.tight_layout()
plt.savefig(results_dir + "AngleResolution-E.png", bbox_inches='tight')


fig, ax = reg.plot_resolution_profile([direction_regression_output_mu], 'direction_errors', towall_binning, x_label="Distance to detector wall in particle direction [cm]", y_label="Direction resolution [°]", y_lim=(0,15), errors=True, x_errors=False)
plt.tight_layout()
plt.savefig(results_dir + "AngleResolution-d.png", bbox_inches='tight')