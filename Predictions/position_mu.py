import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

parser = argparse.ArgumentParser(description="Prediction display script for 3D positions")
parser.add_argument("output_path", type=str,
                    help="Path to the output folder containing outputs/indices.npy and outputs/predicted_positions.npy")
parser.add_argument("classification_path", type=str,
                    help="Path to the npy file containing the classification result")
args = parser.parse_args()

indices_path = args.output_path + "/outputs/indices.npy"
predicted_positions_path = args.output_path + "/outputs/predicted_positions.npy"
os.makedirs(args.output_path + "/results", exist_ok=True)

idxs = np.load(indices_path)                # (M,)
positions = np.load(predicted_positions_path)  # (M, 3)
x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(x, z, y, c=y, cmap="viridis", s=10, alpha=0.7)

ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
ax.set_title("3D positions")
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("Y")

tank_half_height = 271.4235 / 2.0
tank_radius = 307.5926 / 2.0

theta = np.linspace(0, 2*np.pi, 100)
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

out_path = os.path.join(args.output_path, "results/positions_scatter3d.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved 3D scatter to {out_path}")