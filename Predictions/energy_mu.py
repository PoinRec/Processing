import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Prediction display script for direction regression tasks for muons")
parser.add_argument("output_path", type=str, help="Path to the output folder containing outputs/indices.npy and outputs/predicted_energies.npy")
parser.add_argument("classification_path", type=str, help="Path to the npy file containing the classification result")
args = parser.parse_args()

indices_path = args.output_path + "/outputs/indices.npy"
predicted_directions_path = args.output_path + "/outputs/predicted_energies.npy"

os.makedirs(args.output_path + "/results", exist_ok=True)

idxs = np.load(indices_path)   # shape (M,)
energies = np.load(predicted_directions_path)  # shape (M,)

plt.figure(figsize=(6,4))
plt.hist(energies, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
plt.xlabel("Energies (MeV)")
plt.ylabel("Counts")
plt.title("Distribution of energies")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output_path, "results/energies_hist.png"), dpi=300)
plt.close()

print("Saved energies_hist.png to results/")