import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import sys
import argparse

sys.path.append('/home/zhihao/WatChMaL')

data_path = "Data/WCTE_data_fixed/wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"
idxs_path = "/home/zhihao/Data/WCTE_data_fixed/split_list_classification.npz"

parser = argparse.ArgumentParser(description="Evaluation script")
parser.add_argument("-r", "--run_dir", type=str, required=True, help="Path to classification run directory")
parser.add_argument("-e", "--efficiency", type=float, default=0.001, help="Desired muon mis-PID rate for profile plots")
args = parser.parse_args()
classification_run_dir = args.run_dir
results_dir = classification_run_dir + "/results/"

import os
os.makedirs(results_dir, exist_ok=True)

tank_half_height= 271.4235 / 2
tank_radius= 307.5926 / 2

h5_file = h5py.File(data_path, "r")

import analysis.classification as clas
import analysis.utils.binning as bins
import watchmal.utils.math as math

test_idxs  = np.load(idxs_path)['test_idxs']
test_event_labels = np.array(h5_file['labels'])[test_idxs].squeeze()
test_event_energies = np.array(h5_file['energies'])[test_idxs].squeeze()
test_event_angles = np.array(h5_file['angles'])[test_idxs].squeeze()
test_event_positions = np.array(h5_file['positions'])[test_idxs].squeeze()

test_e_energies = test_event_energies[test_event_labels == 1]

classification_output = clas.WatChMaLClassification(classification_run_dir, "ResNet-50 PID", test_event_labels, test_idxs)


fig, ax1, ax2 = classification_output.plot_training_progression(y_loss_lim=(0,10))
plt.tight_layout()
plt.savefig(results_dir + "Accuracy&Loss.png", bbox_inches='tight')


signal_labels = [1] # electrons
background_labels = [2] # muons
clas.plot_rocs([classification_output], signal_labels, background_labels, x_label="Muon PID efficiency (mis-PID rate)", y_label="Electron PID efficiency", mode='efficiency', auc_digits=5)
plt.tight_layout()
plt.savefig(results_dir + "ROC.png", bbox_inches='tight')

desired_muon_efficiency = args.efficiency


with open(results_dir + "ROC_Points.txt", "w") as f:
  pid_cut = classification_output.cut_with_fixed_efficiency(signal_labels, background_labels, desired_muon_efficiency, select_labels=background_labels)

  e_accept = np.mean(pid_cut[test_event_labels==1]) * 100
  mu_accept = np.mean(pid_cut[test_event_labels==2]) * 100

  print(f"\n{e_accept}% of electrons are accepted")
  print(f"{mu_accept}% of muons are accepted\n")

  f.write(f"Muon Rejection Rate: {(1 - desired_muon_efficiency) * 100} %\n")
  f.write(f"{e_accept}% of electrons are accepted\n")
  f.write(f"{mu_accept}% of muons are accepted\n\n")


E_min_val = test_e_energies.min()
E_max_val = test_e_energies.max()
E_binning, E_bin_indices = bins.get_binning(test_event_energies, 20, E_min_val, E_max_val)


clas.plot_efficiency_profile([classification_output], (E_binning, E_bin_indices), select_labels=signal_labels, x_label="True energy [MeV]", y_label="Electron signal PID efficiency [%]", errors=True, x_errors=False, y_lim=(50,100))
plt.tight_layout()
plt.savefig(results_dir + "e-E.png", bbox_inches='tight')


test_event_towall = math.towall(test_event_positions, test_event_angles, tank_half_height=tank_half_height, tank_radius=tank_radius)
test_e_towall = test_event_towall[test_event_labels == 1]

towall_min_val = test_e_towall.min()
towall_max_val = test_e_towall.max()


towall_binning = bins.get_binning(test_event_towall, 20, towall_min_val, towall_max_val)
clas.plot_efficiency_profile([classification_output], towall_binning, select_labels=signal_labels, x_label="Distance to detector wall in particle direction [cm]", y_label="Electron signal PID efficiency [%]", errors=True, x_errors=False, y_lim=(50,100))
plt.tight_layout()
plt.savefig(results_dir + "e-towall.png", bbox_inches='tight')
