import sys, os
import h5py
import numpy as np
from typing import List, Tuple

CHUNK_ROWS = 100_000
COMPRESSION = dict(compression="gzip", compression_opts=4)

def copy_attrs(src_obj, dst_obj):
  for k, v in src_obj.attrs.items():
    dst_obj.attrs[k] = v

def ensure_same_structure(files: List[h5py.File], path="/"):
  ref = set(files[0][path].keys())
  for f in files[1:]:
    if set(f[path].keys()) != ref:
      raise ValueError(f"Structure mismatch at {path}")
  for k in ref:
    if isinstance(files[0][path][k], h5py.Group):
      ensure_same_structure(files, os.path.join(path, k))

def create_or_get_dset(dst_group: h5py.Group, name: str, first_dset: h5py.Dataset):
  shape = first_dset.shape
  dtype = first_dset.dtype

  if name in dst_group:
    return dst_group[name]

  if len(shape) == 0:
    d = dst_group.create_dataset(name, data=first_dset[()], dtype=dtype)
    copy_attrs(first_dset, d)
    return d

  maxshape = (None,) + shape[1:]
  chunks = (min(shape[0] if shape[0] else 1, CHUNK_ROWS),) + shape[1:] if len(shape) > 0 else None
  d = dst_group.create_dataset(
    name,
    shape=(0,) + shape[1:],
    maxshape=maxshape,
    dtype=dtype,
    chunks=chunks,
    **COMPRESSION
  )
  copy_attrs(first_dset, d)
  return d

def append_dataset(dst_dset: h5py.Dataset, src_dset: h5py.Dataset):
  if len(src_dset.shape) == 0:
    return

  if src_dset.dtype != dst_dset.dtype or src_dset.shape[1:] != dst_dset.shape[1:]:
    raise ValueError(f"dtype/shape mismatch for dataset {dst_dset.name}")

  n_old = dst_dset.shape[0]
  n_add = src_dset.shape[0]
  dst_dset.resize((n_old + n_add,) + dst_dset.shape[1:])

  if n_add == 0:
    return

  for start in range(0, n_add, CHUNK_ROWS):
    end = min(start + CHUNK_ROWS, n_add)
    dst_dset[n_old + start : n_old + end, ...] = src_dset[start:end, ...]

def merge_group(dst_group: h5py.Group, src_groups: List[h5py.Group], path="/"):
  copy_attrs(src_groups[0], dst_group)

  for key in src_groups[0].keys():
    objs = [g[key] for g in src_groups]
    if isinstance(objs[0], h5py.Group):
      if key not in dst_group:
        new_grp = dst_group.create_group(key)
      else:
        new_grp = dst_group[key]
      merge_group(new_grp, objs, os.path.join(path, key))
    else:
      dst_dset = create_or_get_dset(dst_group, key, objs[0])
      for ds in objs:
        append_dataset(dst_dset, ds)

def main():
  if len(sys.argv) < 3:
    print("Usage: python merge_h5.py output.h5 input1.h5 input2.h5 ...")
    sys.exit(1)

  out_path = sys.argv[1]
  in_paths = sys.argv[2:]

  files = [h5py.File(p, "r") for p in in_paths]
  try:
    ensure_same_structure(files, "/")

    with h5py.File(out_path, "w") as fout:
      merge_group(fout, [f["/"] for f in files], "/")

    print(f"Merged {len(in_paths)} files -> {out_path}")
  finally:
    for f in files:
      f.close()

if __name__ == "__main__":
  main()