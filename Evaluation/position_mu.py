import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import sys

sys.path.append('/home/zhihao/WatChMaL')

data_path = "/home/zhihao/Data/WCTE_data_fixed/wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"
idxs_path = "/home/zhihao/Data/WCTE_data_fixed/split_list_mu.npz"
regression_run_dir = "/home/zhihao/Data/WCTE_data_fixed/Output/position_mu_2025-07-31_14:19:18"

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
position_regression_output_mu = reg.WatChMaLPositionRegression(regression_run_dir, "muon Regression", test_event_positions, test_event_directions, test_idxs,)


fig, ax = position_regression_output_mu.plot_training_progression()
ax.set_yscale('log')
plt.savefig("Loss.png")


position_resolution = np.quantile(position_regression_output_mu.position_3d_errors, 0.68)
print(f"Overall position resolution (68th percentile of 3D position errors) = {position_resolution} cm")

position_resolution = np.quantile(position_regression_output_mu.position_longitudinal_errors, 0.68)
print(f"Longitudinal position resolution (68th percentile of 3D position errors) = {position_resolution} cm")

position_resolution = np.quantile(position_regression_output_mu.position_transverse_errors, 0.68)
print(f"Transverse position resolution (68th percentile of 3D position errors) = {position_resolution} cm")



E_min_val = test_event_energies.min()
E_max_val = test_event_energies.max()
E_binning = bins.get_binning(test_event_energies, 20, E_min_val, E_max_val)


towall_min_val = test_event_towall.min()
towall_max_val = test_event_towall.max()
towall_binning = bins.get_binning(test_event_towall, 20, towall_min_val, towall_max_val)



fig, ax = reg.plot_resolution_profile([position_regression_output_mu], 'position_3d_errors', E_binning, x_label="True particle energy [MeV]", y_label="Position resolution [cm]", y_lim=(0,15), errors=True, x_errors=False)
plt.savefig("PositionResolution-E.png")

fig, ax = reg.plot_resolution_profile([position_regression_output_mu], 'position_longitudinal_errors', E_binning, x_label="True particle energy [MeV]", y_label="Longitudinal position resolution [cm]", y_lim=(0,15), errors=True, x_errors=False)
plt.savefig("LongitudinalPositionResolution-E.png")

fig, ax = reg.plot_resolution_profile([position_regression_output_mu], 'position_transverse_errors', E_binning, x_label="True particle energy [MeV]", y_label="Transverse position resolution [cm]", y_lim=(0,15), errors=True, x_errors=False)
plt.savefig("TransversePositionResolution-E.png")



fig, ax = reg.plot_resolution_profile([position_regression_output_mu], 'position_3d_errors', towall_binning, x_label="Distance to detector wall in particle direction [cm]", y_label="Position resolution [cm]", y_lim=(0,15), errors=True, x_errors=False)
plt.savefig("PositionResolution-d.png")

fig, ax = reg.plot_resolution_profile([position_regression_output_mu], 'position_longitudinal_errors', towall_binning, x_label="Distance to detector wall in particle direction [cm]", y_label="Longitudinal position resolution [cm]", y_lim=(0,15), errors=True, x_errors=False)
plt.savefig("LongitudinalPositionResolution-d.png")

fig, ax = reg.plot_resolution_profile([position_regression_output_mu], 'position_transverse_errors', towall_binning, x_label="Distance to detector wall in particle direction [cm]", y_label="Transverse position resolution [cm]", y_lim=(0,15), errors=True, x_errors=False)
plt.savefig("TransversePositionResolution-d.png")