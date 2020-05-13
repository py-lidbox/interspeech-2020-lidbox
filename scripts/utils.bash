#!/usr/bin/env bash
set -ue

function print_env {
    echo "==== env begin ===="
    env | sort | uniq
    echo "===== env end ====="
    if [ ! -z "$(command -v module)" ]; then
        module list
    fi
    if [ ! -z "$(command -v nvidia-smi)" ]; then
        nvidia-smi
    fi
    echo
    echo "slurm job nodelist: $SLURM_JOB_NODELIST"
}

function echo_err {
    echo "$@" >> /dev/stderr
}

function get_model_key {
    local config="$1"
    python3 scripts/yaml-get.py $config experiment.model.key sklearn_experiment.model.key 2> /dev/null
}

function get_experiment_key {
    local config="$1"
    python3 scripts/yaml-get.py $config experiment.name sklearn_experiment.name 2> /dev/null
}

function get_first_dataset_key {
    local config="$1"
    python3 scripts/yaml-get.py $config 'datasets[0].key' 2> /dev/null
}

function basedirdirname {
    basename $(dirname $(dirname $1))
}

function pretty_json {
    local jsonpath="$1"
    if [ ! -z $(command -v jq) ]; then
        # pretty-printed json
        jq . $jsonpath
    else
        # fall back to ugly-printed json
        cat $jsonpath
    fi
}

function assert_commands_exist {
    local command_list=("$@")
    local error=0
    for cmd in ${command_list[@]}; do
        if [ -z "$(command -v $cmd)" ]; then
            echo "error: required command '$cmd' not found"
            error=1
        else
            echo "$cmd is $(command -v $cmd)"
        fi
    done
    if [ $error -ne 0 ]; then
        exit $error
    fi
}
