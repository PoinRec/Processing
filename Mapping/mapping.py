import numpy as np

geo_filename = "/home/zhihao/Data/WCTE_data_fixed/geofile_wcte.npz"
mpmt_positions_filename = "/home/zhihao/Data/WCTE_data_fixed/WCTE_mPMT_image_positions_v3.npz"

geo_file = np.load(geo_filename)

pmt_positions = geo_file['position']
pmt_mpmt_id = np.arange(pmt_positions.shape[0])//19
pmt_in_mpmt = np.arange(pmt_positions.shape[0])%19

import matplotlib
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# top down view
axs[0].plot(pmt_positions[:,2], pmt_positions[:,0], 'o', c='lightgray')
axs[0].set_title("Top-down view")
# annotate the mPMTs closest to the bottom
for i in np.where(pmt_positions[:,1] > 50)[0][::19]: #every 19th PMT means we annotate once per PMT
  axs[0].text(pmt_positions[i, 2], pmt_positions[i, 0], pmt_mpmt_id[i])

# unwrapped barrel
phi = np.arctan2(-pmt_positions[:,2], pmt_positions[:,0])
phi[phi < np.pi / 12] += 2 * np.pi # cycle them to positive phi values
axs[1].plot(phi, pmt_positions[:,1], 'o', c='lightgray')
axs[1].set_title("Barrel view")
# Annotate barrel mPMTs
for i in np.where(np.abs(pmt_positions[:,1]) < 120)[0][::19]:
  axs[1].text(phi[i], pmt_positions[i,1], pmt_mpmt_id[i])

# bottom up view
axs[2].plot(pmt_positions[:,2], -pmt_positions[:,0], 'o', c='lightgray')
axs[2].set_title("Bottom-up view")
# annotate the mPMTs closest to the bottom
for i in np.where(pmt_positions[:,1] < -70)[0][::19]: #every 19th PMT means we annotate once per PMT
  axs[2].text(pmt_positions[i, 2], -pmt_positions[i, 0], pmt_mpmt_id[i])
plt.savefig("mPMT_3D.png")

'''
# These are the arrays we will need to fill up
mpmt_image_row = np.empty(len(pmt_positions) // 19, dtype=int)
mpmt_image_column = np.empty(len(pmt_positions) // 19, dtype=int)


# Define the ranges of mPMT IDs in different regions of the detector
top_barrel_mpmts = np.arange(66, 79)
rest_of_barrel_mpmts = np.arange(0, 45)
bottom_cap_mpmts = np.arange(45, 66)
top_cap_mpmts = np.arange(79, 97)


# top endcap: rows 0 to 2 of image
mpmt_image_row[top_cap_mpmts] = (65 - top_cap_mpmts) // 3
# top row of barrel: row 3 of image
mpmt_image_row[top_barrel_mpmts] = 3
# rest of barrel: rows 4 to 6 of image
mpmt_image_row[rest_of_barrel_mpmts] = 4 + rest_of_barrel_mpmts // 12
# bottom endcap: rows 7 to 9 of image
mpmt_image_row[bottom_cap_mpmts] = 7 + (bottom_cap_mpmts - 36) // 3


# top row of barrel spans all 12 columns
mpmt_image_column[top_barrel_mpmts] = 11 - np.arange(12)
# rest of barrel spans all 12 columns
mpmt_image_column[rest_of_barrel_mpmts] = 11 - rest_of_barrel_mpmts%12
# top endcap aligns with the middle of the 12 columns
mpmt_image_column[top_cap_mpmts] = 4 + top_cap_mpmts % 3
# bottom endcap aligns with the middle of the 12 columns
mpmt_image_column[bottom_cap_mpmts] = 4 + bottom_cap_mpmts % 3



mpmt_image_positions = np.column_stack((mpmt_image_row, mpmt_image_column))

plt.plot(mpmt_image_column, -mpmt_image_row, 'o')
for i, p in enumerate(mpmt_image_positions):
  plt.text(p[1], -p[0], i)
plt.savefig("mPMT_2D.png")



fig, axs = plt.subplots(1, 3, figsize=(20,5))

# colored by each coordinate:
for i, coord in enumerate(["x", "y", "z"]):
  scatter = axs[i].scatter(mpmt_image_column, -mpmt_image_row, c=pmt_positions[::19, i])
  axs[i].set_title(f"{coord} coordinate [cm]")
  plt.colorbar(scatter, ax=axs[i])
plt.savefig("mPMT_coord.png")
    
    

np.savez(mpmt_positions_filename, mpmt_image_positions=mpmt_image_positions)
'''