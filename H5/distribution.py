import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Generate various distributions from HDF5 data")
parser.add_argument("data_path", type=str, help="Path to the input HDF5 file")
args = parser.parse_args()
data_path = args.data_path

output_dir = os.path.join(os.path.dirname(data_path), "Distribution")
os.makedirs(output_dir, exist_ok=True)

with h5py.File(data_path, "r") as f:
  angles = np.array(f["angles"][:])            # shape (N, 2)
  energies = np.array(f["energies"][:, 0])     # shape (N,)
  hit_charge = np.array(f["hit_charge"][:])
  hit_time = np.array(f["hit_time"][:])
  labels = np.array(f["labels"][:])
  positions = np.array(f["positions"][:, 0, :])  # shape (N, 3)

x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]

r = np.sqrt(x ** 2 + z ** 2)
theta = np.arctan2(z, x)

cos = np.cos(angles[:, 0])

# Masks for particle types
electron_mask = labels == 1
muon_mask = labels == 2


# === Check label distribution along data index ===
plt.figure(figsize=(12, 4))
plt.plot(labels, marker='.', linestyle='None', markersize=1)
plt.title("Labels distribution along data index")
plt.xlabel("Sample index")
plt.ylabel("Label")
plt.yticks([1, 2], ["Electron", "Muon"])
plt.ylim(0.5, 2.5)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "labels_order_distribution.png"))
plt.close()

# === Check position distribution along data index ===
plt.figure(figsize=(15, 5))
coords = ['r', 'y', 'theta']
data = np.array([r, y, theta])
for i, coord in enumerate(coords):
  plt.subplot(1, 3, i+1)
  plt.plot(data[i], marker='.', linestyle='None', markersize=1)
  plt.title(f"Position {coord} along data index")
  plt.xlabel("Sample index")
  plt.ylabel(f"Position {coord} [cm]" if coord != 'theta' else "theta [rad]")
  plt.grid(True)


plt.tight_layout()
plt.savefig(os.path.join(output_dir, "position_order_distribution.png"))
plt.close()


# === Angle distribution along data index ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cos, marker='.', linestyle='None', markersize=1)
plt.title("cos(theta) distribution along data index")
plt.xlabel("Sample index")
plt.ylabel("cos(theta)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(angles[:, 1], marker='.', linestyle='None', markersize=1)
plt.title("Phi angle distribution along data index")
plt.xlabel("Sample index")
plt.ylabel("Phi [rad]")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "angle_order_distribution.png"))
plt.close()


# Energy Distribution
plt.figure()
plt.hist(energies[electron_mask], bins=100, alpha=0.7, label="Electron")
plt.hist(energies[muon_mask], bins=100, alpha=0.7, label="Muon")
plt.xlabel("Energy [MeV]")
plt.ylabel("Counts")
plt.title("Energy Distribution by Particle Type")
plt.legend()
plt.savefig(os.path.join(output_dir, "energy_distribution.png"))
plt.close()


# Angular Distribution
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.hist(cos[electron_mask], bins=100, alpha=0.7, label="Electron")
plt.hist(cos[muon_mask], bins=100, alpha=0.7, label="Muon")
plt.xlabel("cos(theta)")
plt.ylabel("Counts")
plt.title("cos(theta) Distribution")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(angles[electron_mask, 1], bins=100, alpha=0.7, label="Electron")
plt.hist(angles[muon_mask, 1], bins=100, alpha=0.7, label="Muon")
plt.xlabel("phi")
plt.ylabel("Counts")
plt.title("phi Distribution")
plt.legend()

plt.savefig(os.path.join(output_dir, "angle_distribution.png"))
plt.close()


# Position Distribution(r, theta, z)
plt.figure(figsize=(15,4))
for i, axis in enumerate(['r', 'y', 'theta']):
  plt.subplot(1, 3, i + 1)
  plt.hist(data[i][electron_mask], bins=100, alpha=0.7, label="Electron")
  plt.hist(data[i][muon_mask], bins=100, alpha=0.7, label="Muon")
  plt.xlabel(f"Position {axis} [cm]" if axis != 'theta' else "theta [rad]")
  plt.ylabel("Counts")
  plt.legend()
plt.suptitle("Position Distribution")
plt.savefig(os.path.join(output_dir, "position_distribution.png"))
plt.close()


# hit_charge, hit_time Distribution
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.hist(hit_charge, bins=100, log=True)
plt.xlabel("Hit Charge")
plt.ylabel("Counts (log scale)")
plt.title("Hit Charge Distribution")

plt.subplot(1, 2, 2)
plt.hist(hit_time, bins=100, log=True)
plt.xlabel("Hit Time [ns]")
plt.ylabel("Counts (log scale)")
plt.title("Hit Time Distribution")

plt.savefig(os.path.join(output_dir, "hit_charge_time_distribution.png"))
plt.close()