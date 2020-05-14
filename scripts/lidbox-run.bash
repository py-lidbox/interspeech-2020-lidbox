#!/usr/bin/env bash
set -ue
subcommand=$1
config=$2
lidbox $subcommand $config -v
