import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(
  description="Generate various distributions from HDF5 data")
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
  fully_contained = np.array(f["fully_contained"][:])

x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]

r = np.sqrt(x ** 2 + z ** 2)
theta = np.arctan2(z, x)

cos = np.cos(angles[:, 0])


# Particle label mapping and masks
# 0 = Gamma, 1 = Electron, 2 = Muon, 3 = Pion
label_map = {0: "Gamma", 1: "Electron", 2: "Muon", 3: "Pion"}
present_codes = sorted(int(c) for c in np.unique(labels))
present_map = {c: label_map.get(int(c), f"Label{int(c)}") for c in present_codes}
masks = {present_map[c]: (labels == c) for c in present_codes}


# === Check label distribution along data index ===
plt.figure(figsize=(12, 4))
plt.plot(labels, marker='.', linestyle='None', markersize=1)
plt.title("Labels distribution along data index")
plt.xlabel("Sample index")
plt.ylabel("Label")
plt.yticks(present_codes, [present_map[c] for c in present_codes])
plt.ylim(min(present_codes) - 0.5, max(present_codes) + 0.5)
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
for label, mask in masks.items():
  plt.hist(energies[mask], bins=100, alpha=0.7, label=label)
plt.xlabel("Energy [MeV]")
plt.ylabel("Counts")
plt.title("Energy Distribution by Particle Type")
plt.legend()
plt.savefig(os.path.join(output_dir, "energy_distribution.png"))
plt.close()


# Angular Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for label, mask in masks.items():
  plt.hist(cos[mask], bins=100, alpha=0.7, label=label)
plt.xlabel("cos(theta)")
plt.ylabel("Counts")
plt.title("cos(theta) Distribution")
plt.legend()

plt.subplot(1, 2, 2)
for label, mask in masks.items():
  plt.hist(angles[mask, 1], bins=100, alpha=0.7, label=label)
plt.xlabel("phi")
plt.ylabel("Counts")
plt.title("phi Distribution")
plt.legend()

plt.savefig(os.path.join(output_dir, "angle_distribution.png"))
plt.close()


# Position Distribution(r, theta, z)
plt.figure(figsize=(15, 4))
for i, axis in enumerate(['r', 'y', 'theta']):
  plt.subplot(1, 3, i + 1)
  for label, mask in masks.items():
    plt.hist(data[i][mask], bins=100, alpha=0.7, label=label)
  plt.xlabel(f"Position {axis} [cm]" if axis != 'theta' else "theta [rad]")
  plt.ylabel("Counts")
  plt.legend()
plt.suptitle("Position Distribution")
plt.savefig(os.path.join(output_dir, "position_distribution.png"))
plt.close()


# hit_charge, hit_time Distribution
plt.figure(figsize=(12, 5))
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


# === Per-particle FC statistics vs Energy (all particles in one plot) ===
energy_bins = np.linspace(energies.min(), energies.max(), 30)
fig, ax_ratio = plt.subplots()
ax_count = ax_ratio.twinx()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
K = max(1, len(masks))
left_edges = energy_bins[:-1]
bin_widths = np.diff(energy_bins)
centers_all = left_edges + bin_widths/2

for i, (label, mask) in enumerate(masks.items()):
  E = energies[mask]
  FC = fully_contained[mask].astype(bool)

  # Use histogram counts for FC
  color = colors[i % len(colors)]
  ax_count.hist(E[FC], bins=energy_bins, alpha=0.5, color=color, label=None, zorder=1)

  # Compute ratio using total vs FC in the same bins
  total_counts, _ = np.histogram(E, bins=energy_bins)
  fc_counts, _ = np.histogram(E[FC], bins=energy_bins)
  ratio = np.divide(fc_counts, total_counts, out=np.full_like(fc_counts, np.nan, dtype=float), where=total_counts > 0)

  valid = total_counts > 0
  ax_ratio.plot(centers_all[valid], ratio[valid], marker='o', linestyle='-', color=color, label=f"{label} Ratio", zorder=5)

ax_ratio.set_zorder(ax_count.get_zorder() + 1)
ax_ratio.patch.set_visible(False)

total_fc_counts = {label: fully_contained[mask].sum() for label, mask in masks.items()}

legend = ax_ratio.legend(loc='upper right', framealpha=0.8, fancybox=True)

text_str = "FC totals:\n" + "\n".join([f"{label}: {count}" for label, count in total_fc_counts.items()])
# Estimate a y-position just below the legend, keeping right alignment
n_legend_lines = max(1, len(legend.get_texts()))
y_pos = 1.0 - 0.06 * n_legend_lines - 0.04   # tweak spacing under legend
ax_ratio.text(0.98, y_pos, text_str, transform=ax_ratio.transAxes, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7'))

ax_count.set_ylabel('FC Event Count')
ax_count.yaxis.set_label_position('right')
ax_count.yaxis.tick_right()

ax_ratio.set_ylim(0, 1.05)
ax_ratio.set_ylabel('FC Ratio')
ax_ratio.set_xlabel('Energy [MeV]')
ax_ratio.set_title('Fully-contained statistics vs Energy')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'FC_vs_energy.png'))
plt.close()

# === Fully-contained order distribution ===
plt.figure(figsize=(12, 4))
plt.plot(fully_contained.astype(int), marker='.', linestyle='None', markersize=1)
plt.yticks([0, 1], ["False", "True"])
plt.xlabel("Sample index")
plt.ylabel("fully_contained")
plt.title("Fully-contained distribution along data index")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "FC_order_distribution.png"))
plt.close()
