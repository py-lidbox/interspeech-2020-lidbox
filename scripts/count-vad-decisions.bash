#!/usr/bin/env bash
set -ue

experiment_dir=
source scripts/env.bash
source scripts/utils.bash

config_files=(
models/ap19-olr/config.prepare.yaml
models/mgb3/config.prepare.yaml
models/dosl/config.prepare.yaml
)
config_files=( "${config_files[@]/#/$experiment_dir/}" )

for config in ${config_files[*]}; do
    jobname=$(get_first_dataset_key $config)-count-vad-decisions
    sbatch \
        --job-name=$jobname \
        --output=$experiment_dir/logs/${jobname}.out \
        --error=$experiment_dir/logs/${jobname}.err \
        --cpus-per-task=5 \
        --time=00-04 \
        --mem=20G \
        $experiment_dir/scripts/lidbox-run.bash utils $config -vv --run-script scripts/compute_dataset_stats.py
done
