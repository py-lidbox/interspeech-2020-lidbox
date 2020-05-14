#!/usr/bin/env bash
# Train all backend models on language vectors, extracted from end-to-end models trained on all datasets
set -ue

experiment_dir=
source scripts/env.bash
source scripts/utils.bash

config_files=(
models/ap19-olr/config.ap19olr-baseline-combined3backendNB.yaml
models/ap19-olr/config.mgb3-baseline-combined3backendNB.yaml
models/ap19-olr/config.dosl-baseline-combined3backendNB.yaml
models/ap19-olr/config.spherespeaker-combined3backendNB.yaml
models/ap19-olr/config.xvec-2d-combined3backendNB.yaml
models/ap19-olr/config.xvec-channeldropout-combined3backendNB.yaml
models/ap19-olr/config.xvec-extended-combined3backendNB.yaml
models/mgb3/config.ap19olr-baseline-combined3backendNB.yaml
models/mgb3/config.mgb3-baseline-combined3backendNB.yaml
models/mgb3/config.dosl-baseline-combined3backendNB.yaml
models/mgb3/config.spherespeaker-combined3backendNB.yaml
models/mgb3/config.xvec-2d-combined3backendNB.yaml
models/mgb3/config.xvec-channeldropout-combined3backendNB.yaml
models/mgb3/config.xvec-extended-combined3backendNB.yaml
models/dosl/config.ap19olr-baseline-combined3backendNB.yaml
models/dosl/config.mgb3-baseline-combined3backendNB.yaml
models/dosl/config.dosl-baseline-combined3backendNB.yaml
models/dosl/config.spherespeaker-combined3backendNB.yaml
models/dosl/config.xvec-2d-combined3backendNB.yaml
models/dosl/config.xvec-channeldropout-combined3backendNB.yaml
models/dosl/config.xvec-extended-combined3backendNB.yaml
)
config_files=( "${config_files[@]/#/$experiment_dir/}" )

echo 'launching open task language vector back-end training tasks on slurm'

for config in ${config_files[*]}; do
    jobname=$(config_to_jobname $config)-train
    sbatch \
        --job-name=$jobname \
        --output=$experiment_dir/logs/${jobname}.out \
        --error=$experiment_dir/logs/${jobname}.err \
        --time=00-04 \
        --gres=gpu:1 \
        --mem=20G \
        $experiment_dir/scripts/lidbox-run.bash train-embeddings $config -v
done
