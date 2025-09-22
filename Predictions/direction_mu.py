import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

sys.path.append('/home/zhihao/WatChMaL')
import watchmal.utils.math as math

parser = argparse.ArgumentParser(description="Prediction display script for direction regression tasks for muons")
parser.add_argument("output_path", type=str, help="Path to the output folder containing outputs/indices.npy and outputs/predicted_directions.npy")
parser.add_argument("classification_path", type=str, help="Path to the npy file containing the classification result")
args = parser.parse_args()

indices_path = args.output_path + "/outputs/indices.npy"
predicted_directions_path = args.output_path + "/outputs/predicted_directions.npy"

os.makedirs(args.output_path + "/results", exist_ok=True)

idxs = np.load(indices_path)   # shape (M,)
directions = np.load(predicted_directions_path)  # shape (M, 3)

angles = math.angles_from_direction(directions)  # shape (M, 2), in degrees


theta = np.degrees(angles[:, 0])
phi   = np.degrees(angles[:, 1])

plt.figure(figsize=(6,4))
plt.hist(theta, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
plt.xlabel("Theta (degrees)")
plt.ylabel("Counts")
plt.title("Distribution of theta")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output_path, "results/theta_hist.png"), dpi=300)
plt.close()

plt.figure(figsize=(6,4))
plt.hist(phi, bins=50, color="salmon", edgecolor="black", alpha=0.7)
plt.xlabel("Phi (degrees)")
plt.ylabel("Counts")
plt.title("Distribution of phi")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output_path, "results/phi_hist.png"), dpi=300)
plt.close()

print("Saved theta_hist.png and phi_hist.png to results/")