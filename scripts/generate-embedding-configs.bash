#!/usr/bin/env bash
# Autogenerate lidbox config files for training Gaussian Naive Bayes
set -ue

source scripts/utils.bash

if [ $# -ne 1 ]; then
    echo_err "error: wrong number of args: $# out of 1"
    echo_err 'usage: bash generate-embedding-configs.bash $(pwd)'
    exit 2
fi
expdir=$1

closed_set_configs=(
models/ap19-olr/config.baseline.yaml
models/ap19-olr/config.sbs-baseline.yaml
models/ap19-olr/config.mgb3-baseline.yaml
models/ap19-olr/config.xvec-channeldropout.yaml
models/ap19-olr/config.xvec-extended.yaml
models/ap19-olr/config.xvec-2d.yaml
models/ap19-olr/config.spherespeaker.yaml
models/mgb3/config.baseline.yaml
models/mgb3/config.sbs-baseline.yaml
models/mgb3/config.ap19olr-baseline.yaml
models/mgb3/config.xvec-channeldropout.yaml
models/mgb3/config.xvec-extended.yaml
models/mgb3/config.xvec-2d.yaml
models/mgb3/config.spherespeaker.yaml
models/sbs/config.baseline.yaml
models/sbs/config.mgb3-baseline.yaml
models/sbs/config.ap19olr-baseline.yaml
models/sbs/config.xvec-channeldropout.yaml
models/sbs/config.xvec-extended.yaml
models/sbs/config.xvec-2d.yaml
models/sbs/config.spherespeaker.yaml
)

open_set_configs=(
models/combined3/config.ap19olr-baseline.yaml
models/combined3/config.sbs-baseline.yaml
models/combined3/config.mgb3-baseline.yaml
models/combined3/config.xvec-channeldropout.yaml
models/combined3/config.xvec-extended.yaml
models/combined3/config.xvec-2d.yaml
models/combined3/config.spherespeaker.yaml
)
dataset_keys=(ap19-olr mgb3 sbs)

echo 'checking closed set config files with lidbox JSON schema'
for cf in ${closed_set_configs[*]}; do
    lidbox utils -v --validate-config-file $expdir/$cf || exit $?
done
echo 'checking open set config files with lidbox JSON schema'
for cf in ${open_set_configs[*]}; do
    lidbox utils -v --validate-config-file $expdir/$cf || exit $?
done
echo 'converting closed set config files'
for cf in ${closed_set_configs[*]}; do
    dataset=$(basename $(dirname $cf))
    out_cf=$(echo $cf | sed 's|\.yaml$|-backendNB.yaml|g')
    python3 $expdir/scripts/generate-embedding-configs.py \
        $expdir/$cf \
        $expdir/$cf \
        $expdir/$out_cf \
        $dataset \
        --batch-size 500 \
        || exit $?
    echo "ok '$cf' -> '$out_cf'"
done
echo 'converting open set config files'
for cf in ${open_set_configs[*]}; do
    for dataset in ${dataset_keys[*]}; do
        out_cf=$(basename $cf | sed 's|\.yaml$|-combined3backendNB.yaml|g')
        out_cf=models/$dataset/$out_cf
        python3 $expdir/scripts/generate-embedding-configs.py \
            $expdir/$cf \
            $expdir/models/$dataset/config.baseline.yaml \
            $expdir/$out_cf \
            $dataset \
            --batch-size 500 \
            || exit $?
        echo "ok '$cf' -> '$out_cf'"
    done
done
