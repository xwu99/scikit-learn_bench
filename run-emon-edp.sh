#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

source /opt/intel/sep/sep_vars.sh

emon -collect-edp -w -f 0_emon-results/emon.dat ./profile.sh
