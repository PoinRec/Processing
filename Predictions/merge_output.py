import numpy as np
import h5py
import argparse
import os
import sys

parser =argparse.ArgumentParser(description="Merge all the npy/npz outputs into a single HDF5 file")
parser.add_argument("output_path", type=str, help="Path to the output folder containing classification and regression outputs")
args = parser.parse_args()

output_h5_path = os.path.join(args.output_path, "merged_outputs.h5")

sys.path.append('/home/zhihao/Processing/Predictions')
import infer_exit_pos

with h5py.File(output_h5_path, "w") as f_out:
  canonical_sorted_indices = None
  canonical_from = None

  for subfolder in ["classification", "direction_mu", "energy_mu", "FC_mu", "position_mu"]:
    subfolder_path = os.path.join(args.output_path, subfolder, "outputs")
    if not os.path.exists(subfolder_path):
      continue

    idx_path = os.path.join(subfolder_path, "indices.npy")
    if not os.path.exists(idx_path):
      continue
    idxs = np.load(idx_path)
    _, order = np.unique(idxs, return_index=True)
    sorted_indices = idxs[order]

    f_out.create_dataset(f"{subfolder}_sorted_indices", data=sorted_indices)

    if canonical_sorted_indices is None:
      canonical_sorted_indices = sorted_indices
      canonical_from = subfolder
      f_out.create_dataset("sorted_indices", data=canonical_sorted_indices)
      f_out["sorted_indices"].attrs["from_subfolder"] = canonical_from
    else:
      if len(sorted_indices) != len(canonical_sorted_indices):
        raise ValueError(
          f"[Index length mismatch] '{subfolder}' has {len(sorted_indices)} "
          f"while canonical '{canonical_from}' has {len(canonical_sorted_indices)}."
        )
      unequal_mask = (sorted_indices != canonical_sorted_indices)
      if np.any(unequal_mask):
        bad_pos = np.nonzero(unequal_mask)[0]
        preview_n = min(10, bad_pos.size)
        pos_preview = bad_pos[:preview_n]
        left_preview = sorted_indices[pos_preview]
        right_preview = canonical_sorted_indices[pos_preview]
        raise ValueError(
          f"[Index content mismatch] '{subfolder}' vs canonical '{canonical_from}': "
          f"{bad_pos.size} differing positions. "
          f"Examples at positions {pos_preview.tolist()} -> "
          f"{subfolder}={left_preview.tolist()} vs canonical={right_preview.tolist()}"
        )

    for file_name in os.listdir(subfolder_path):
      if file_name == "indices.npy":
        continue
      if file_name.endswith(".npy"):
        file_path = os.path.join(subfolder_path, file_name)
        data = np.load(file_path)
        data = data[order, ...]  # reorder according to indices
        ds_name = f"{subfolder}_{file_name.removesuffix('.npy')}"
        f_out.create_dataset(ds_name, data=data)

  f_out.attrs["indices_consistency_checked"] = True
             
        
POS_KEY = "position_mu_predicted_positions"
DIR_KEY = "direction_mu_predicted_directions"
   

with h5py.File(output_h5_path, "r+") as f_out:
  if POS_KEY not in f_out or DIR_KEY not in f_out:
    raise KeyError(f"{POS_KEY} / {DIR_KEY} not found in {output_h5_path}")

  positions  = f_out[POS_KEY][...]   # (N, 3)
  directions = f_out[DIR_KEY][...]   # (N, 3)

  result = infer_exit_pos.ray_cylinder_intersection_yaxis_batch(positions, directions)
  
  if isinstance(result, dict):
    t_exit, X_exit, kind_exit = result["t"], result["X"], result["kind"]
  else:
    t_exit, X_exit, kind_exit = result  # 假设是 (t, X, kind)

  def upsert(name, data, *, dtype=None):
    if name in f_out:
      del f_out[name]
    if dtype is not None:
      f_out.create_dataset(name, data=data, dtype=dtype)
    else:
      f_out.create_dataset(name, data=data)

  upsert("exit_points", X_exit)               # (N, 3)
  upsert("exit_t", t_exit)                    # (N,)

  str_dtype = h5py.string_dtype(encoding="utf-8")
  upsert("exit_kind", np.asarray(kind_exit, dtype=object), dtype=str_dtype)

  hit_mask = np.isfinite(t_exit)
  counts = hit_mask.sum()
  print(f"Computed exit points for {counts} / {len(t_exit)} rays ({counts / len(t_exit) * 100:.2f}%)")
  upsert("exit_hit_mask", hit_mask.astype(np.uint8))
  

      