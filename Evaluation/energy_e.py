import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import h5py
import sys

sys.path.append('/home/zhihao/WatChMaL')

import analysis.regression as reg
import analysis.utils.binning as bins
import watchmal.utils.math as math


data_path = "/home/zhihao/Data/WCTE_data_fixed/wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"
idxs_path = "/home/zhihao/Data/WCTE_data_fixed/split_list_e.npz"
regression_run_dir = "/home/zhihao/Data/WCTE_data_fixed/Output/energy_e_larger_lr"


h5_file = h5py.File(data_path, "r")
test_idxs = np.load(idxs_path)['test_idxs']

test_event_labels = np.array(h5_file['labels'])[test_idxs].squeeze()
test_event_energies = np.array(h5_file['energies'])[test_idxs].squeeze()
test_event_angles = np.array(h5_file['angles'])[test_idxs].squeeze()
test_event_positions = np.array(h5_file['positions'])[test_idxs].squeeze()

test_event_momenta = math.momentum_from_energy(test_event_energies, test_event_labels)

tank_half_height = 271.4235 / 2
tank_radius = 307.5926 / 2
test_event_towall = math.towall(test_event_positions, test_event_angles, tank_half_height=tank_half_height, tank_radius=tank_radius)
test_event_directions = math.direction_from_angles(test_event_angles)



energy_regression_output_e = reg.WatChMaLEnergyRegression(
  regression_run_dir,
  "Electron Energy Regression",
  true_momenta=test_event_momenta,
  true_labels=test_event_labels,
  indices=test_idxs,
)


if hasattr(energy_regression_output_e, 'plot_training_progression'):
  fig, ax = energy_regression_output_e.plot_training_progression()
  ax.set_yscale('log')
  plt.savefig("EnergyLoss.png")



momentum_fractional_errors = energy_regression_output_e.momentum_fractional_errors
momentum_resolution = np.quantile(np.abs(momentum_fractional_errors), 0.68)

print(f"Overall momentum resolution (68th percentile of momentum fractional errors) = {momentum_resolution * 100:.1f} %")

momentum_residuals = energy_regression_output_e.momentum_residuals
momentum_resolution_r = np.quantile(np.abs(momentum_residuals), 0.68)

print(f"Overall momentum resolution (68th percentile of momentum residuals) = {momentum_resolution_r:.1f} MeV")

print("momentum_fractional_errors min/max:", np.min(energy_regression_output_e.momentum_fractional_errors), np.max(energy_regression_output_e.momentum_fractional_errors))
print("momentum_fractional_errors has NaN:", np.isnan(energy_regression_output_e.momentum_fractional_errors).any())

'''
plt.figure(figsize=(8,5))
plt.hist(test_event_energies, bins=50, color='blue', alpha=0.7)
plt.title('Electron Energy Distribution in Test Set')
plt.xlabel('True Electron Energy [MeV]')
plt.ylabel('Number of Samples')
plt.grid(True)
plt.tight_layout()
plt.savefig("ElectronEnergyDistribution.png")
plt.close()

plt.figure(figsize=(8,5))
plt.hist(test_event_towall, bins=50, color='blue', alpha=0.7)
plt.title('Electron Distance to Detector Wall Distribution')
plt.xlabel('Distance to Wall [cm]')
plt.ylabel('Number of Samples')
plt.grid(True)
plt.tight_layout()
plt.savefig("ElectronDistanceToWallDistribution.png")
plt.close()
'''


E_min_val = test_event_energies.min()
E_max_val = test_event_energies.max()
E_binning = bins.get_binning(test_event_energies, 20, E_min_val, E_max_val)


towall_min_val = test_event_towall.min()
towall_max_val = test_event_towall.max()
towall_binning = bins.get_binning(test_event_towall, 20, towall_min_val, towall_max_val)


fig, ax = reg.plot_resolution_profile(
  [energy_regression_output_e],
  'momentum_fractional_errors',
  E_binning,
  x_label="True particle energy [MeV]",
  y_label="Momentum resolution [%]",
  y_lim=(0, momentum_resolution * 3),
  errors=True,
  x_errors=False
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.savefig("MomentumResolution_vs_Energy.png")


fig, ax = reg.plot_resolution_profile(
  [energy_regression_output_e],
  'momentum_fractional_errors',
  towall_binning,
  x_label="Distance to detector wall in particle direction [cm]",
  y_label="Momentum resolution [%]",
  y_lim=(0, momentum_resolution * 3),
  errors=True,
  x_errors=False
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.savefig("MomentumResolution_vs_Distance.png")


fig, ax = reg.plot_bias_profile(
  [energy_regression_output_e],
  'momentum_fractional_errors',
  E_binning,
  x_label="True particle energy [MeV]",
  y_label="Momentum bias [%]",
  y_lim=(-0.15, 0.15),
  errors=True,
  x_errors=False
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.savefig("MomentumBias_vs_Energy.png")


fig, ax = reg.plot_bias_profile(
  [energy_regression_output_e],
  'momentum_fractional_errors',
  towall_binning,
  x_label="Distance to detector wall in particle direction [cm]",
  y_label="Momentum bias [%]",
  y_lim=(-0.15, 0.15),
  errors=True,
  x_errors=False
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.savefig("MomentumBias_vs_Distance.png")


