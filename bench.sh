#!/usr/bin/env bash

source /home/xiaochang/miniconda3/bin/activate xgb15-conda-default

# CONFIG_FILES=configs_benchmark/xgb_pandas_F_spr.json
# CONFIG_FILES=configs_benchmark/xgb_santander_spr.json
# CONFIG_FILES=configs_benchmark/xgb_letters_spr.json
# CONFIG_FILES=configs_benchmark/xgb_gpu_spr.json
# CONFIG_FILES=configs_benchmark/xgb_gpu_spr_scale_prediction.json
# CONFIG_FILES=configs_benchmark/xgb_cpu_additional_config_spr.json
# CONFIG_FILES=configs_benchmark/xgb_epsilon_spr.json
CONFIG_FILES=configs_profile/xgb_higgs1m_numpy_C.json

BASE_NAME=$(basename -s .json $CONFIG_FILES)_${CONDA_DEFAULT_ENV}
mkdir -p 0_results/${BASE_NAME}/config
mkdir -p 0_results/${BASE_NAME}/result
cp ${CONFIG_FILES} 0_results/${BASE_NAME}/config

for run_no in {1..3}
do
OUTPUT_FILE=0_results/${BASE_NAME}/result/${BASE_NAME}_run${run_no}.json
LOG_FILE=0_results/${BASE_NAME}/result/${BASE_NAME}_run${run_no}.log
python runner.py --configs $CONFIG_FILES --output-file $OUTPUT_FILE | tee $LOG_FILE
done
