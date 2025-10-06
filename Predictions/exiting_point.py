import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os


tank_half_height = 271.4235 / 2.0
tank_radius = 307.5926 / 2.0


parser = argparse.ArgumentParser(description="Prediction display script for exiting points for muons")
parser.add_argument("merged_output_path", type=str, help="Path to the merged_outputs.h5")
args = parser.parse_args()

result_path = os.path.dirname(args.merged_output_path) + "/merged_results"
os.makedirs(result_path, exist_ok=True)

with h5py.File(args.merged_output_path, "r") as f:
  FC_mu_softmax = np.array(f["FC_mu_softmax"][:])  # shape (N, 2)
  classification_softmax = np.array(f["classification_softmax"][:])  # shape (N, 2)
  directions = np.array(f["direction_mu_predicted_directions"][:])  # shape (N, 3)
  energies = np.array(f["energy_mu_predicted_energies"]).squeeze()  # shape (N, 1)
  positions = np.array(f["position_mu_predicted_positions"][:])  # shape (N, 3)
  exit_hit_mask = np.array(f["exit_hit_mask"][:]).astype(bool)  # shape (N,)
  exit_kind = np.array(f["exit_kind"][:])  # shape (N,) str
  exit_t = np.array(f["exit_t"][:])  # shape (N,)
  exit_pos = np.array(f["exit_points"][:])  # shape (N, 3)


x = exit_pos[:, 0]
y = exit_pos[:, 1]
z = exit_pos[:, 2]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(x, z, y, c=y, cmap="viridis", s=10, alpha=0.7)

ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
ax.set_title("3D positions")
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Y")


theta = np.linspace(0, 2 * np.pi, 100)
yy = np.linspace(-tank_half_height, tank_half_height, 60)
Theta, YY = np.meshgrid(theta, yy)

XC = tank_radius * np.cos(Theta)
ZC = tank_radius * np.sin(Theta)
YC = YY

ax.plot_surface(XC, ZC, YC, rstride=1, cstride=1, color='lightgray', alpha=0.15, edgecolor='none')

theta_dense = np.linspace(0, 2*np.pi, 200)
xc = tank_radius * np.cos(theta_dense)
zc = tank_radius * np.sin(theta_dense)
ax.plot(xc, zc, np.full_like(xc, tank_half_height), color='gray', linewidth=1, alpha=0.6)
ax.plot(xc, zc, np.full_like(xc, -tank_half_height), color='gray', linewidth=1, alpha=0.6)

pad = 0.1 * tank_radius
x_min, x_max = -tank_radius - pad, tank_radius + pad
z_min, z_max = -tank_radius - pad, tank_radius + pad
y_min, y_max = -tank_half_height - pad, tank_half_height + pad
ax.set_xlim(x_min, x_max)
ax.set_ylim(z_min, z_max)
ax.set_zlim(y_min, y_max)

ax.set_box_aspect((x_max - x_min, z_max - z_min, y_max - y_min))

out_path = os.path.join(result_path, "positions_scatter3d.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved 3D scatter to {out_path}")