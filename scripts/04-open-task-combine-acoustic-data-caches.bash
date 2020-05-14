#!/usr/bin/env bash
# Combine the signal caches gathered in step 1
set -ue
experiment_dir=
source scripts/env.bash
jobname=combine-signal-caches-for-open-task
sbatch \
    --job-name=$jobname \
    --output=$experiment_dir/logs/${jobname}.out \
    --error=$experiment_dir/logs/${jobname}.err \
    --cpus-per-task=5 \
    --time=00-04 \
    --mem=60G \
    $experiment_dir/scripts/combine-acoustic-data-caches.py $experiment_dir
