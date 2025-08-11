import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Check mPMT mapping continuity")
parser.add_argument("geo_filename", type=str, help="Path to the geometry .npz file")
parser.add_argument("mpmt_positions_filename", type=str, help="Path to the mPMT image positions .npz file")
args = parser.parse_args()

geo_filename = args.geo_filename
mpmt_positions_filename = args.mpmt_positions_filename

output_dir = os.path.join(os.path.dirname(geo_filename), "Checking")
os.makedirs(output_dir, exist_ok=True)

geo_file = np.load(geo_filename)
pmt_positions = geo_file['position']


mpmt_positions_file = np.load(mpmt_positions_filename)
keys = mpmt_positions_file.files

mpmt_image_positions = mpmt_positions_file[keys[0]]

plt.plot(mpmt_image_positions[:, 1], -mpmt_image_positions[:, 0], 'o')
for i, p in enumerate(mpmt_image_positions):
  plt.text(p[1], -p[0], i)
plt.savefig(os.path.join(output_dir, "mPMT_2D.png"))


fig, axs = plt.subplots(1, 3, figsize=(20,5))

# colored by each coordinate:
for i, coord in enumerate(["x", "y", "z"]):
  scatter = axs[i].scatter(mpmt_image_positions[:, 1], -mpmt_image_positions[:, 0], c=pmt_positions[::19, i])
  axs[i].set_title(f"{coord} coordinate [cm]")
  plt.colorbar(scatter, ax=axs[i])
plt.savefig(os.path.join(output_dir, "mPMT_coord.png"))