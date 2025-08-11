import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="Check PMT orientation from geometry file")
parser.add_argument("geo_filename", type=str, help="Path to the geometry .npz file")
args = parser.parse_args()

geo_filename = args.geo_filename
geo = np.load(geo_filename)

output_dir = os.path.join(os.path.dirname(geo_filename), "Checking")
os.makedirs(output_dir, exist_ok=True)

positions = geo['position']  # shape: (1843, 3)
channels = np.arange(positions.shape[0]) % 19  # channel ID within each mPMT

'''
for i in [45,46,47,53,54,55,56,57,64,65,81,82,83,84,89,90,91,92,93,94]:
  base = i * 19
  # Swap positions according to the specified pattern
  positions[base + 1, :], positions[base + 4, :] = positions[base + 4, :].copy(), positions[base + 1, :].copy()
  positions[base + 2, :], positions[base + 5, :] = positions[base + 5, :].copy(), positions[base + 2, :].copy()
  positions[base + 3, :], positions[base + 6, :] = positions[base + 6, :].copy(), positions[base + 3, :].copy()
  positions[base + 7, :], positions[base + 13, :] = positions[base + 13, :].copy(), positions[base + 7, :].copy()
  positions[base + 8, :], positions[base + 14, :] = positions[base + 14, :].copy(), positions[base + 8, :].copy()
  positions[base + 9, :], positions[base + 15, :] = positions[base + 15, :].copy(), positions[base + 9, :].copy()
  positions[base + 10, :], positions[base + 16, :] = positions[base + 16, :].copy(), positions[base + 10, :].copy()
  positions[base + 11, :], positions[base + 17, :] = positions[base + 17, :].copy(), positions[base + 11, :].copy()
  positions[base + 12, :], positions[base + 18, :] = positions[base + 18, :].copy(), positions[base + 12, :].copy()

'''


x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot colored by PMT channel ID
sc = ax.scatter(x, y, z, c=channels, cmap='tab20', s=1)
fig.colorbar(sc, ax=ax, label='PMT Channel (0-18)')
ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_zlabel('Z [cm]')
ax.set_title('3D PMT Positions (colored by channel ID)')
plt.savefig(os.path.join(output_dir, '3D_PMT_Positions_colored_by_channel_ID.png'))

# Apply mask: only keep PMTs where y > 100
mask = y > 100
x_proj = x[mask]
z_proj = z[mask]
channels_proj = channels[mask]

# Plot x-z projection for y > 100
plt.figure()
plt.scatter(x_proj, z_proj, c=channels_proj, cmap='tab20', s=5)
plt.xlabel('X [cm]')
plt.ylabel('Z [cm]')
plt.title('XZ Projection of PMTs (y > 100)')
plt.colorbar(label='PMT Channel (0-18)')
plt.axis('equal')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'XZ_Projection_PMT_Positions_y_gt_100.png'))


# Apply mask: only keep PMTs where y < -100
mask = y < -100
x_proj = x[mask]
z_proj = z[mask]
channels_proj = channels[mask]

# Plot x-z projection for y < -100
plt.figure()
plt.scatter(x_proj, z_proj, c=channels_proj, cmap='tab20', s=5)
plt.xlabel('X [cm]')
plt.ylabel('Z [cm]')
plt.title('XZ Projection of PMTs (y < -100)')
plt.colorbar(label='PMT Channel (0-18)')
plt.axis('equal')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'XZ_Projection_PMT_Positions_y_lt_-100.png'))