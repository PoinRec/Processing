import numpy as np

# Load the PMT position array from the npz file
# position has shape (1843, 3), each row is (x, y, z) coordinate of a PMT
position = np.load('/home/zhihao/Data/WCTE_data_fixed/geofile_wcte.npz')['position']

# Specify the tank axis index (usually 1 means y-axis)
tank_axis = 1

# Extract PMT coordinates along the tank axis (e.g., y-coordinate)
pos_along = position[:, tank_axis]

# Extract PMT coordinates perpendicular to the tank axis (remove the axis dimension)
pos_trans = np.delete(position, tank_axis, axis=1)

# Calculate the detector height as the difference between max and min positions along the axis
height = pos_along.max() - pos_along.min()
print(f"Detector height ≈ {height:.2f} units")

# Calculate the detector radius as the maximum distance from the axis in the transverse plane
radius = np.linalg.norm(pos_trans, axis=1).max()
print(f"Detector radius ≈ {radius:.2f} units")