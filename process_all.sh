#!/bin/bash

set -euo pipefail

read -r -p "data path: " DATA_PATH
read -r -p "geometry path: " GEO_PATH
read -r -p "image position path: " POS_PATH
read -r -p "Event number you want to check: " EVENT_NUM

[[ -e $DATA_PATH ]] || { echo "DATA_PATH not found"; exit 1; }
[[ -e $GEO_PATH  ]] || { echo "GEO_PATH not found";  exit 1; }
[[ -e $POS_PATH  ]] || { echo "POS_PATH not found";  exit 1; }


singularity exec --nv -B ~:/home/zhihao ~/Images/larcv2_ub2204-cuda121-torch251-larndsim-2025-03-20.sif bash -lc "

  cd /home/zhihao/Processing
  
  cd MappingCheck
  python continuity_check.py \"$GEO_PATH\" \"$POS_PATH\"
  python orientation_check.py \"$GEO_PATH\"
  python viewing.py \"$GEO_PATH\" \"$POS_PATH\"

  cd ../H5
  python distribution.py \"$DATA_PATH\"
  python FCcheck.py \"$DATA_PATH\"
  if [ -n \"$EVENT_NUM\" ]; then
    python viewing.py \"$DATA_PATH\" \"$POS_PATH\" \"$GEO_PATH\" \"$EVENT_NUM\"
  else
    python viewing.py \"$DATA_PATH\" \"$POS_PATH\" \"$GEO_PATH\"
  fi

"
