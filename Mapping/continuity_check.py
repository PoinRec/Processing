import numpy as np
import matplotlib.pyplot as plt

geo_filename = "/home/zhihao/Data/WCTE_data_fixed/geofile_wcte.npz"
mpmt_positions_filename = "/home/zhihao/Data/WCTE_data_fixed/WCTE_mPMT_image_positions_v3.npz"


geo_file = np.load(geo_filename)
pmt_positions = geo_file['position']


mpmt_positions_file = np.load(mpmt_positions_filename)
keys = mpmt_positions_file.files

mpmt_image_positions = mpmt_positions_file[keys[0]]

plt.plot(mpmt_image_positions[:, 1], -mpmt_image_positions[:, 0], 'o')
for i, p in enumerate(mpmt_image_positions):
  plt.text(p[1], -p[0], i)
plt.savefig("mPMT_2D.png")


fig, axs = plt.subplots(1, 3, figsize=(20,5))

# colored by each coordinate:
for i, coord in enumerate(["x", "y", "z"]):
  scatter = axs[i].scatter(mpmt_image_positions[:, 1], -mpmt_image_positions[:, 0], c=pmt_positions[::19, i])
  axs[i].set_title(f"{coord} coordinate [cm]")
  plt.colorbar(scatter, ax=axs[i])
plt.savefig("mPMT_coord.png")