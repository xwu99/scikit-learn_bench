#!/usr/bin/env bash

source /home/xiaochang/miniconda3/bin/activate dev

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

source /opt/intel/sep/sep_vars.sh

cd 0_emon-results

emon -process-pyedp ../pyedp_config.txt
