#!/usr/bin/env bash
# Launch batches of lidbox jobs on slurm with sbatch
set -ue

source scripts/utils.bash

# Load config file list and remove comment lines
# mapfile -t config_files < scripts/lidbox-config.list
# config_files=(${config_files[@]##\#*})
config_files=(
# models/ap19-olr/config.baseline-adam-eps1.yaml
# models/ap19-olr/config.baseline-adam-lr0001-eps1.yaml
models/ap19-olr/config.spherespeaker-embed1000.yaml
models/mgb3/config.spherespeaker-embed1000.yaml
models/dosl/config.spherespeaker-embed1000.yaml
models/combined3/config.spherespeaker-embed1000.yaml
)

num_prepare_cpus=5
slurm_kwargs="\
    --constraint=avx \
    "

verbosity='-v'

function check_env_errors {
    local expdir="$1"
    local cf="$2"
    local config=$expdir/$cf
    local error=0
    if [ -z $(command -v lidbox) ]; then
        echo_err "error: command 'lidbox' not found, did you install the package?"
        error=1
    fi
    if [ ! -f $config ]; then
        echo_err "error: lidbox config file '$config' does not exist"
        error=1
    fi
    local lb_model_key=$(get_model_key $config)
    local checkpoint_dir=$(dirname $config)/cache/$lb_model_key/checkpoints
    if [ -d "$checkpoint_dir" ]; then
        if [ $(find $checkpoint_dir -type f | wc -l) -gt 0 ]; then
            echo_err "warning: model trained using config file '$config' already has keras checkpoints in '$checkpoint_dir', training might be resumed from a checkpoint"
        fi
    fi
    local logdir=$(dirname $checkpoint_dir)/tensorboard/logs
    if [ -d $logdir ]; then
        local num_logdirs=$(echo $(ls $logdir) | wc -l)
        if [ $num_logdirs -gt 0 ]; then
            echo_err "warning: there are already $num_logdirs tensorboard logs at '$logdir'"
        fi
    fi
    local features_dir=$(dirname $config)/cache/dataset
    if [ ! -d "$features_dir" ]; then
        echo_err "warning: there are no features in '$features_dir', they might be extracted before training can begin (unless you are using a non-persistent file cache)"
    fi
    return $error
}

function usage {
    echo_err 'usage: batch-train.bash experiment_dir jobtype flag'
}

if [ $# -ne 3 ]; then
    echo_err "error: wrong number of args: $# out of 3"
    usage
    exit 2
fi
expdir=$1
jobtype=$2
dryrun=false
case $3 in
    dryrun)
        dryrun=true
        echo 'flag is dryrun, will not call slurm'
        ;;
    sbatch)
        echo 'flag is sbatch, will launch slurm tasks'
        ;;
    *)
        echo_err "error: unknown flag '$3'"
        exit 2
        ;;
esac
case $jobtype in
    train*)
        echo "jobtype is train ($jobtype), will use gpu"
        ;;
    prepare*)
        echo "jobtype is prepare ($jobtype), will use $num_prepare_cpus cpus"
        ;;
    evaluate*)
        echo "jobtype is evaluate ($jobtype), will use gpu"
        ;;
    *)
        echo_err "error: unknown jobtype '$jobtype'"
        exit 2
        ;;
esac

echo 'running pre-check for all config files'
for cf in ${config_files[*]}; do
    echo 'checking config file with lidbox JSON schema'
    lidbox utils -v --validate-config-file $cf || exit $?
    check_env_errors $expdir $cf || exit $?
    echo "ok '$cf'"
done

for cf in ${config_files[*]}; do
    config=$expdir/$cf
    lb_experiment_key=$(get_experiment_key $config)
    lb_model_key=$(get_model_key $config)
    jobname_base=lb-${lb_model_key}-${lb_experiment_key}
    case $jobtype in
        train)
            jobname=${jobname_base}-train
            cmd="\
sbatch \
$slurm_kwargs \
--job-name=$jobname \
--output=$expdir/logs/${jobname}.out \
--error=$expdir/logs/${jobname}.err \
--cpus-per-task=2 \
--time=01-00 \
--gres=gpu:1 \
--mem=32G \
$expdir/scripts/lidbox-run.bash $config e2e $verbosity"
            ;;
        train-long)
            jobname=${jobname_base}-train
            cmd="\
sbatch \
$slurm_kwargs \
--job-name=$jobname \
--output=$expdir/logs/${jobname}.out \
--error=$expdir/logs/${jobname}.err \
--constraint=volta \
--time=04-00 \
--gres=gpu:1 \
--mem=32G \
$expdir/scripts/lidbox-run.bash $config e2e $verbosity"
            ;;
        train-short)
            jobname=${jobname_base}-train
            cmd="\
sbatch \
$slurm_kwargs \
--job-name=$jobname \
--output=$expdir/logs/${jobname}.out \
--error=$expdir/logs/${jobname}.err \
--time=00-04 \
--gres=gpu:1 \
--mem=32G \
$expdir/scripts/lidbox-run.bash $config e2e $verbosity"
            ;;
        train-short-embed)
            jobname=${jobname_base}-train
            cmd="\
sbatch \
$slurm_kwargs \
--job-name=$jobname \
--output=$expdir/logs/${jobname}.out \
--error=$expdir/logs/${jobname}.err \
--gres=gpu:1 \
--time=00-04 \
--mem=32G \
$expdir/scripts/lidbox-run.bash $config train-embeddings $verbosity"
            ;;
        evaluate)
            jobname=${jobname_base}-evaluate
            cmd="\
sbatch \
$slurm_kwargs \
--job-name=$jobname \
--output=$expdir/logs/${jobname}.out \
--error=$expdir/logs/${jobname}.err \
--constraint=volta \
--time=00-04 \
--gres=gpu:1 \
--mem=32G \
$expdir/scripts/lidbox-run.bash $config evaluate $verbosity"
            ;;
        prepare-long)
            jobname=${jobname_base}-prepare
            cmd="\
sbatch \
$slurm_kwargs \
--job-name=$jobname \
--output=$expdir/logs/${jobname}.out \
--error=$expdir/logs/${jobname}.err \
--time=04-00 \
--cpus-per-task=$num_prepare_cpus \
--mem=64G \
$expdir/scripts/lidbox-run.bash $config prepare $verbosity"
            ;;
        prepare)
            jobname=${jobname_base}-prepare
            cmd="\
sbatch \
$slurm_kwargs \
--job-name=$jobname \
--output=$expdir/logs/${jobname}.out \
--error=$expdir/logs/${jobname}.err \
--time=01-00 \
--cpus-per-task=$num_prepare_cpus \
--mem=64G \
$expdir/scripts/lidbox-run.bash $config prepare $verbosity"
            ;;
        prepare-short)
            jobname=${jobname_base}-prepare
            cmd="\
sbatch \
$slurm_kwargs \
--job-name=$jobname \
--output=$expdir/logs/${jobname}.out \
--error=$expdir/logs/${jobname}.err \
--time=00-04 \
--cpus-per-task=$num_prepare_cpus \
--mem=64G \
$expdir/scripts/lidbox-run.bash $config prepare $verbosity"
            ;;
        prepare-gpu)
            jobname=${jobname_base}-prepare
            cmd="\
sbatch \
$slurm_kwargs \
--job-name=$jobname \
--output=$expdir/logs/${jobname}.out \
--error=$expdir/logs/${jobname}.err \
--gres=gpu:1 \
--time=00-04 \
--cpus-per-task=2 \
--mem=32G \
$expdir/scripts/lidbox-run.bash $config prepare $verbosity"
            ;;
            *)
            echo "unknown jobtype '$jobtype'"
            exit 1
            ;;
    esac
    error=0
    for logfile in "$expdir/logs/${jobname}.out" "$expdir/logs/${jobname}.err"; do
        if [ -f "$logfile" ]; then
            printf "error, logfile '$logfile' already exists, overwrite existing logfile? (y/n) "
            read -n 1 ans
            echo
            if [ "$ans" == y ]; then
                error=0
            else
                error=1
            fi
        fi
    done
    if [ $error -ne 0 ]; then
        exit $error
    fi
    if $dryrun; then
        printf "DRYRUN: $(echo $cmd | tr ' ' '\n')\n\n"
    else
        $cmd || exit $?
    fi
done
