#!/usr/bin/env bash
# Autogenerate lidbox config files for training Gaussian Naive Bayes
set -ue

experiment_dir=
source scripts/env.bash
source scripts/utils.bash

closed_set_configs=(
models/ap19-olr/config.ap19olr-baseline.yaml
models/ap19-olr/config.dosl-baseline.yaml
models/ap19-olr/config.mgb3-baseline.yaml
models/ap19-olr/config.spherespeaker.yaml
models/ap19-olr/config.xvec-2d.yaml
models/ap19-olr/config.xvec-channeldropout.yaml
models/ap19-olr/config.xvec-extended.yaml
models/dosl/config.ap19olr-baseline.yaml
models/dosl/config.dosl-baseline.yaml
models/dosl/config.mgb3-baseline.yaml
models/dosl/config.spherespeaker.yaml
models/dosl/config.xvec-2d.yaml
models/dosl/config.xvec-channeldropout.yaml
models/dosl/config.xvec-extended.yaml
models/mgb3/config.ap19olr-baseline.yaml
models/mgb3/config.dosl-baseline.yaml
models/mgb3/config.mgb3-baseline.yaml
models/mgb3/config.spherespeaker.yaml
models/mgb3/config.xvec-2d.yaml
models/mgb3/config.xvec-channeldropout.yaml
models/mgb3/config.xvec-extended.yaml
)

open_set_configs=(
models/combined3/config.ap19olr-baseline.yaml
models/combined3/config.dosl-baseline.yaml
models/combined3/config.mgb3-baseline.yaml
models/combined3/config.spherespeaker.yaml
models/combined3/config.xvec-2d.yaml
models/combined3/config.xvec-channeldropout.yaml
models/combined3/config.xvec-extended.yaml
)
dataset_keys=(ap19-olr mgb3 dosl)

echo 'checking closed set config files with lidbox JSON schema'
for cf in ${closed_set_configs[*]}; do
    lidbox utils -v --validate-config-file $experiment_dir/$cf || exit $?
done
echo 'checking open set config files with lidbox JSON schema'
for cf in ${open_set_configs[*]}; do
    lidbox utils -v --validate-config-file $experiment_dir/$cf || exit $?
done
echo 'converting closed set config files'
for cf in ${closed_set_configs[*]}; do
    dataset=$(basename $(dirname $cf))
    out_cf=$(echo $cf | sed 's|\.yaml$|-backendNB.yaml|g')
    python3 $experiment_dir/scripts/generate_embedding_config.py \
        $experiment_dir/$cf \
        $experiment_dir/$cf \
        $experiment_dir/$out_cf \
        $dataset \
        --batch-size 100 \
        || exit $?
    echo "ok '$cf' -> '$out_cf'"
done
echo 'converting open set config files'
for cf in ${open_set_configs[*]}; do
    for dataset in ${dataset_keys[*]}; do
        out_cf=$(basename $cf | sed 's|\.yaml$|-combined3backendNB.yaml|g')
        out_cf=models/$dataset/$out_cf
        python3 $experiment_dir/scripts/generate_embedding_config.py \
            $experiment_dir/$cf \
            $experiment_dir/models/$dataset/config.$(echo $dataset | tr --delete '-')-baseline.yaml \
            $experiment_dir/$out_cf \
            $dataset \
            --batch-size 100 \
            || exit $?
        echo "ok '$cf' -> '$out_cf'"
    done
done
