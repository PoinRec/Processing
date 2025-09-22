import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

from matplotlib.lines import Line2D
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="Prediction display script for classification tasks")
parser.add_argument("output_path", type=str, help="Path to the output folder containing outputs/indices.npy and outputs/softmax.npy")
args = parser.parse_args()

indices_path = args.output_path + "/outputs/indices.npy"
softmax_path = args.output_path + "/outputs/softmax.npy"

os.makedirs(args.output_path + "/results", exist_ok=True)

idxs = np.load(indices_path)   # shape (M,)
probs = np.load(softmax_path)  # shape (M, C)

counts = np.zeros(probs.shape[1], dtype=int)
for i in range(len(idxs)):
  counts[probs[i].argmax()] += 1
print("Top-1 counts (with duplicates):", counts)

eps = 1e-20
ratio = (probs[:, 1] + eps) / (probs[:, 0] + eps)

uniq_ids, first_pos = np.unique(idxs, return_index=True)
idxs_first  = idxs[first_pos]
ratio_first = ratio[first_pos]

mask_first = ratio_first < 1
e_indices = idxs_first[mask_first]
e_values  = ratio_first[mask_first]
mu_indices = idxs_first[~mask_first]
mu_values  = ratio_first[~mask_first]

e_order = np.argsort(e_indices)
e_indices = e_indices[e_order]
e_values  = e_values[e_order]

mu_order = np.argsort(mu_indices)
mu_indices = mu_indices[mu_order]
mu_values  = mu_values[mu_order]

plt.figure(figsize=(8, 6))

ratio_clipped = np.clip(ratio, eps, None)
log_ratio = np.log10(ratio_clipped)

norm = TwoSlopeNorm(vmin=log_ratio.min(), vcenter=0.0, vmax=log_ratio.max())

sc = plt.scatter(
  idxs, ratio,
  c=log_ratio,
  cmap="coolwarm",
  norm=norm,
  s=10, alpha=0.8
)

plt.xlabel("Sample index")
plt.ylabel("Ratio (log scale)")
plt.yscale("log")
plt.title("Scatter plot of ratio across samples")
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(sc)
cbar.set_label("Ratio value (log-colored, white at 1)")

log_min = np.floor(log_ratio.min())
log_max = np.ceil(log_ratio.max())
ticks = np.arange(log_min, log_max + 1)

def fmt_func(v, pos):
  if abs(v) < 1e-12:
    return "1"
  return f"1e{int(v)}"
cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt_func))

cmap = plt.get_cmap("coolwarm")

rep_low_color  = cmap(norm(np.log10(1e-6)))
rep_high_color = cmap(norm(np.log10(1e6)))

legend_ge1 = Line2D([0],[0], marker='o', linestyle='none', markerfacecolor=rep_high_color, markeredgecolor='k', markersize=7, label=f"ratio ≥ 1 ({len(mu_indices)})")

legend_lt1 = Line2D([0],[0], marker='o', linestyle='none', markerfacecolor=rep_low_color, markeredgecolor='k', markersize=7, label=f"ratio < 1 ({len(e_indices)})")

plt.legend(handles=[legend_ge1, legend_lt1], loc="best")

plt.savefig(args.output_path + "/results/ratio_scatter.png", dpi=300)

print(f"{len(e_indices)} unique events with ratio < 1")
print("indices & ratio values (first occurrence, sorted by index):")
for i, (idx, val) in enumerate(zip(e_indices, e_values)):
  print(f"{i:4d}: index={idx}, ratio={val}")

np.savez(
  args.output_path + "/results/classification_result.npz",
  e_idxs=e_indices, e_vals=e_values,
  mu_idxs=mu_indices, mu_vals=mu_values
)