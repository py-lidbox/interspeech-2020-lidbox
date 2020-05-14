#!/usr/bin/env bash
# Train all non-baseline models on three datasets
set -ue

experiment_dir=
source scripts/env.bash
source scripts/utils.bash

# We skip the baseline models since they were trained in step 02
config_files=(
# models/ap19-olr/config.ap19olr-baseline.yaml
models/ap19-olr/config.mgb3-baseline.yaml
models/ap19-olr/config.dosl-baseline.yaml
models/ap19-olr/config.spherespeaker.yaml
models/ap19-olr/config.xvec-2d.yaml
models/ap19-olr/config.xvec-channeldropout.yaml
models/ap19-olr/config.xvec-extended.yaml
models/mgb3/config.ap19olr-baseline.yaml
# models/mgb3/config.mgb3-baseline.yaml
models/mgb3/config.dosl-baseline.yaml
models/mgb3/config.spherespeaker.yaml
models/mgb3/config.xvec-2d.yaml
models/mgb3/config.xvec-channeldropout.yaml
models/mgb3/config.xvec-extended.yaml
models/dosl/config.ap19olr-baseline.yaml
models/dosl/config.mgb3-baseline.yaml
# models/dosl/config.dosl-baseline.yaml
models/dosl/config.spherespeaker.yaml
models/dosl/config.xvec-2d.yaml
models/dosl/config.xvec-channeldropout.yaml
models/dosl/config.xvec-extended.yaml
)
config_files=( "${config_files[@]/#/$experiment_dir/}" )

echo 'launching training tasks on slurm'

for config in ${config_files[*]}; do
    jobname=$(config_to_jobname $config)-train
    sbatch \
        --job-name=$jobname \
        --output=$experiment_dir/logs/${jobname}.out \
        --error=$experiment_dir/logs/${jobname}.err \
        --time=01-00 \
        --gres=gpu:1 \
        --mem=20G \
        $experiment_dir/scripts/lidbox-run.bash e2e $config
done
