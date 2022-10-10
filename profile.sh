#!/usr/bin/env bash

source /home/xiaochang/miniconda3/bin/activate xgb15-conda-default

CONFIG_FILES=configs_profile_skl/linear_10M_20.json

BASE_NAME=$(basename -s .json $CONFIG_FILES)_${CONDA_DEFAULT_ENV}
TIMESTAMP=$(date +%m%d_%H%M%S)
mkdir -p 0_results/${BASE_NAME}

OUTPUT_FILE=0_results/${BASE_NAME}/${BASE_NAME}_${TIMESTAMP}.json
LOG_FILE=0_results/${BASE_NAME}/${BASE_NAME}_${TIMESTAMP}.log

# python runner.py --configs $CONFIG_FILES --output-file $OUTPUT_FILE | tee $LOG_FILE

python sklearn_bench/linear_profile.py --arch sr256 --data-format numpy --data-order C --dtype float64 --num-threads -1 --time-limit 120 --test-num-threads -1 --box-filter-measurements 20000 --device none --file-X-train data/synthetic-regression-X-train-10000000x20.npy --file-y-train data/synthetic-regression-y-train-10000000x20.npy --file-X-test data/synthetic-regression-X-train-10000000x20.npy --file-y-test data/synthetic-regression-y-train-10000000x20.npy --dataset-name synthetic_regression
