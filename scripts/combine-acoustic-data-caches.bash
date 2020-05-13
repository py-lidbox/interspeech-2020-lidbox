#!/usr/bin/env bash
# Wrapper code for the Aalto compute cluster
set -ue
experiment_dir='/m/triton/scratch/elec/puhe/p/lindgrm1/exp'
module load cuda/10.1.243
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WRKDIR}/lib/cuda/lib64
export PYTHONUNBUFFERED=1
python3 scripts/combine-acoustic-data-caches.py $experiment_dir
