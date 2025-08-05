import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import sys

sys.path.append('/home/zhihao/WatChMaL')

data_path = "Data/WCTE_data_fixed/wcte_CDS_pgun_e-_3M_mu-_3M_0to1GeV_fixedFC.h5"
idxs_path = "/home/zhihao/Data/WCTE_data_fixed/split_list_classification.npz"
classification_run_dir = "/home/zhihao/Data/WCTE_data_fixed/Output/classification_00"

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
plt.savefig("Accuracy&Loss.png")


signal_labels = [1] # electrons
background_labels = [2] # muons
clas.plot_rocs([classification_output], signal_labels, background_labels, x_label="Muon PID efficiency (mis-PID rate)", y_label="Electron PID efficiency", mode='efficiency')
plt.savefig("ROC.png")




desired_muon_efficiency = 0.1
pid_cut = classification_output.cut_with_fixed_efficiency(signal_labels, background_labels, desired_muon_efficiency, select_labels=background_labels)

print(f"{np.mean(pid_cut[test_event_labels==1])*100}% of electrons are accepted")
print(f"{np.mean(pid_cut[test_event_labels==2])*100}% of muons are accepted")


E_min_val = test_e_energies.min()
E_max_val = test_e_energies.max()
E_binning, E_bin_indices = bins.get_binning(test_event_energies, 20, E_min_val, E_max_val)


clas.plot_efficiency_profile([classification_output], (E_binning, E_bin_indices), select_labels=signal_labels, x_label="True energy [MeV]", y_label="Electron signal PID efficiency [%]", errors=True, x_errors=False, y_lim=(90,100))
plt.show()
plt.savefig("e-E.png")


test_event_towall = math.towall(test_event_positions, test_event_angles, tank_half_height=tank_half_height, tank_radius=tank_radius)
test_e_towall = test_event_towall[test_event_labels == 1]

towall_min_val = test_e_towall.min()
towall_max_val = test_e_towall.max()


towall_binning = bins.get_binning(test_event_towall, 20, towall_min_val, towall_max_val)
clas.plot_efficiency_profile([classification_output], towall_binning, select_labels=signal_labels, x_label="Distance to detector wall in particle direction [cm]", y_label="Electron signal PID efficiency [%]", errors=True, x_errors=False, y_lim=(90,100))
plt.show()
plt.savefig("e-towall.png")
