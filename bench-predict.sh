#!/usr/bin/env bash

source /home/xiaochang/miniconda3/bin/activate xgb15-conda-default

export PYTHONPATH=${PWD}
NUM_THREADS=$(nproc)

BASE_NAME=optimized_prediction
OUTPUT_DIR=0_results/${BASE_NAME}

mkdir -p ${OUTPUT_DIR}

for run_no in {1..3}
do

python -W ignore xgboost_bench/gbt_predict_optimized.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --device none --file-X-train data/airline-ohe_x_train.npy --file-y-train data/airline-ohe_y_train.npy --file-X-test data/airline-ohe_10x_tile_x_test.npy --file-y-test data/airline-ohe_10x_tile_y_test.npy --dataset-name airline-ohe | tee ${OUTPUT_DIR}/airline-ohe-${BASE_NAME}-run${run_no}.log

python -W ignore xgboost_bench/gbt_predict_optimized.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_100x_tile_x_test.npy --file-y-test data/higgs1m_100x_tile_y_test.npy --dataset-name higgs1m | tee ${OUTPUT_DIR}/higgs1m-${BASE_NAME}-run${run_no}.log

python -W ignore xgboost_bench/gbt_predict_optimized.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --max-bin 256 --learning-rate 0.3 --subsample 1 --reg-lambda 2 --min-child-weight 1 --min-split-loss 0.1 --max-depth 8 --n-estimators 200 --objective multi:softprob --device none --file-X-train data/mlsr_3x_tile_x_test.npy --file-y-train data/mlsr_3x_tile_y_test.npy --dataset-name mlsr | tee ${OUTPUT_DIR}/mslr-${BASE_NAME}-run${run_no}.log

done