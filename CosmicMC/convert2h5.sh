#!/bin/bash
set -euo pipefail
shopt -s nullglob

mkdir -p "$HOME/Data/WCTE_cosmic_mc/wcte_cosmics_mc_fiTQun/h5"

singularity exec --nv -B ~:/home/zhihao \
  ~/Images/larcv2_ub2204-cuda121-torch251-larndsim-2025-03-20.sif \
  bash -lc '

    ROOT_DIR="/home/zhihao/Data/WCTE_cosmic_mc/wcte_cosmics_mc_fiTQun"
    OUT_DIR="/home/zhihao/Data/WCTE_cosmic_mc/wcte_cosmics_mc_fiTQun/h5"
    MERGED="/home/zhihao/Data/WCTE_cosmic_mc/merged_fq.h5"
    PROC="/home/zhihao/Processing"

    mkdir -p "$OUT_DIR"

    for file in "$ROOT_DIR"/*.root; do
      echo "Processing $file"
      base_name="$(basename "$file" .root)"
      python "$PROC/fiTQun/extract_from_fq.py" \
        "$file" \
        "$OUT_DIR/${base_name}.h5" \
        --no-require-rpc
    done

    h5_list=( "$OUT_DIR"/*.h5 )
    if ((${#h5_list[@]} == 0)); then
      echo "No HDF5 files in $OUT_DIR to merge. Exit."
      exit 1
    fi

    python "$PROC/CosmicMC/mergeh5.py" \
      "$MERGED" \
      "${h5_list[@]}"

    echo "Merged -> $MERGED"
'