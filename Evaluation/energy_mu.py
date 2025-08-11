import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import h5py
import sys
import argparse

sys.path.append('/home/zhihao/WatChMaL')

import analysis.regression as reg
import analysis.utils.binning as bins
import watchmal.utils.math as math


data_path = "/home/zhihao/Data/WCTE_data_fixed/wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"
idxs_path = "/home/zhihao/Data/WCTE_data_fixed/Splitting/split_list_mu.npz"

parser = argparse.ArgumentParser(description="Evaluation script")
parser.add_argument("run_dir", type=str, help="Path to muon energy regression run directory")
args = parser.parse_args()
regression_run_dir = args.run_dir
results_dir = regression_run_dir + "/results/"

import os
os.makedirs(results_dir, exist_ok=True)

h5_file = h5py.File(data_path, "r")
test_idxs = np.load(idxs_path)['test_idxs']
train_idxs = np.load(idxs_path)['train_idxs']

test_event_labels = np.array(h5_file['labels'])[test_idxs].squeeze()
train_event_labels = np.array(h5_file['labels'])[train_idxs].squeeze()
test_event_energies = np.array(h5_file['energies'])[test_idxs].squeeze()
train_event_energies = np.array(h5_file['energies'])[train_idxs].squeeze()
test_event_angles = np.array(h5_file['angles'])[test_idxs].squeeze()
test_event_positions = np.array(h5_file['positions'])[test_idxs].squeeze()

test_event_momenta = math.momentum_from_energy(test_event_energies, test_event_labels)

tank_half_height = 271.4235 / 2
tank_radius = 307.5926 / 2
test_event_towall = math.towall(test_event_positions, test_event_angles, tank_half_height=tank_half_height, tank_radius=tank_radius)
test_event_directions = math.direction_from_angles(test_event_angles)

selection_mu_train = (train_event_labels == 2)

energy_regression_output_mu = reg.WatChMaLEnergyRegression(
  regression_run_dir,
  "Muon Energy Regression",
  true_momenta=test_event_momenta,
  true_labels=test_event_labels,
  indices=test_idxs,
)


fig, ax = energy_regression_output_mu.plot_training_progression()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(results_dir + "EnergyLoss.png", bbox_inches='tight')


momentum_fractional_errors = energy_regression_output_mu.momentum_fractional_errors
momentum_resolution = np.quantile(np.abs(momentum_fractional_errors), 0.68)
momentum_fractional_bias = np.mean(momentum_fractional_errors)


momentum_residuals = energy_regression_output_mu.momentum_residuals
momentum_resolution_r = np.quantile(np.abs(momentum_residuals), 0.68)
momentum_bias = np.mean(momentum_residuals)


with open(results_dir + "resolution.txt", "w") as f:
  print(f"Overall momentum resolution (68th percentile of momentum fractional errors) = {momentum_resolution * 100:.2f} %")
  f.write(f"Overall momentum resolution (68th percentile of momentum fractional errors) = {momentum_resolution * 100:.2f} %\n")
  
  print(f"Overall momentum resolution (68th percentile of momentum residuals) = {momentum_resolution_r:.2f} MeV")
  f.write(f"Overall momentum resolution (68th percentile of momentum residuals) = {momentum_resolution_r:.2f} MeV\n")
  
  print(f"Overall momentum fractional bias = {momentum_fractional_bias * 100:.2f} %")
  f.write(f"Overall momentum fractional bias = {momentum_fractional_bias * 100:.2f} %\n")
  
  print(f"Overall momentum bias = {momentum_bias:.2f} MeV")
  f.write(f"Overall momentum bias = {momentum_bias:.2f} MeV\n")
  
  
  print("momentum_fractional_errors min/max:", np.min(energy_regression_output_mu.momentum_fractional_errors), np.max(energy_regression_output_mu.momentum_fractional_errors))
  print("momentum_fractional_errors has NaN:", np.isnan(energy_regression_output_mu.momentum_fractional_errors).any())
  
  f.write(f"momentum_fractional_errors min/max: {np.min(energy_regression_output_mu.momentum_fractional_errors)}, {np.max(energy_regression_output_mu.momentum_fractional_errors)}\n")
  f.write(f"momentum_fractional_errors has NaN: {np.isnan(energy_regression_output_mu.momentum_fractional_errors).any()}\n")

'''
plt.figure(figsize=(8,5))
plt.hist(test_event_energies, bins=50, color='orange', alpha=0.7)
plt.title('Muon Energy Distribution in Test Set')
plt.xlabel('True Muon Energy [MeV]')
plt.ylabel('Number of Samples')
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir + "MuonEnergyDistribution_test.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
plt.hist(train_event_energies[selection_mu_train], bins=50, color='orange', alpha=0.7)
plt.title('Muon Energy Distribution in Train Set')
plt.xlabel('True Muon Energy [MeV]')
plt.ylabel('Number of Samples')
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir + "MuonEnergyDistribution_train.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
plt.hist(test_event_towall, bins=50, color='orange', alpha=0.7)
plt.title('Muon Distance to Detector Wall Distribution')
plt.xlabel('Distance to Wall [cm]')
plt.ylabel('Number of Samples')
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir + "MuonDistanceToWallDistribution.png", bbox_inches='tight')
plt.close()
'''


E_min_val_mu = test_event_energies.min()
E_max_val_mu = test_event_energies.max()
E_binning_mu = bins.get_binning(test_event_energies, 20, E_min_val_mu, E_max_val_mu)

  
towall_min_val_mu = test_event_towall.min()
towall_max_val_mu = test_event_towall.max()
towall_binning_mu = bins.get_binning(test_event_towall, 20, towall_min_val_mu, towall_max_val_mu)


fig, ax = reg.plot_resolution_profile(
  [energy_regression_output_mu],
  'momentum_fractional_errors',
  E_binning_mu,
  x_label="True muon energy [MeV]",
  y_label="Muon momentum resolution [%]",
  y_lim=(0, 0.5),
  errors=True,
  x_errors=False
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.tight_layout()
plt.savefig(results_dir + "MuonMomentumResolution_vs_Energy.png", bbox_inches='tight')


fig, ax = reg.plot_resolution_profile(
  [energy_regression_output_mu],
  'momentum_fractional_errors',
  towall_binning_mu,
  x_label="Distance to detector wall in muon direction [cm]",
  y_label="Muon momentum resolution [%]",
  y_lim=(0, momentum_resolution * 3),
  errors=True,
  x_errors=False
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.tight_layout()
plt.savefig(results_dir + "MuonMomentumResolution_vs_Distance.png", bbox_inches='tight')


fig, ax = reg.plot_bias_profile(
  [energy_regression_output_mu],
  'momentum_fractional_errors',
  E_binning_mu,
  x_label="True muon energy [MeV]",
  y_label="Muon momentum bias [%]",
  y_lim=(-0.50, 0.15),
  errors=True,
  x_errors=False
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.tight_layout()
plt.savefig(results_dir + "MuonMomentumBias_vs_Energy.png", bbox_inches='tight')


fig, ax = reg.plot_bias_profile(
  [energy_regression_output_mu],
  'momentum_fractional_errors',
  towall_binning_mu,
  x_label="Distance to detector wall in muon direction [cm]",
  y_label="Muon momentum bias [%]",
  y_lim=(-0.05, 0.05),
  errors=True,
  x_errors=False
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.tight_layout()
plt.savefig(results_dir + "MuonMomentumBias_vs_Distance.png", bbox_inches='tight')