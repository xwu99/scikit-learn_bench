#!/usr/bin/env bash

# source /home/xiaochang/miniconda3/bin/activate xgb15-conda-default
# source /home/xiaochang/miniconda3/bin/activate xgb12
source /home/xiaochang/miniconda3/bin/activate xgb-daal4py

export PYTHONPATH=${PWD}
NUM_THREADS=$(nproc)

# higgs1m shape: Train: (1000000, 28), Test: (500000, 28)
python -W ignore xgboost_bench/gbt_predict_optimized.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy --dataset-name higgs1m

# higgs1m w/ scaled test dataset shape: Train: (1000000, 28), Test: (50000000, 28)
# python -W ignore xgboost_bench/gbt_predict_optimized.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_100x_tile_x_test.npy --file-y-test data/higgs1m_100x_tile_y_test.npy --dataset-name higgs1m

# airline-ohe shape: Train: (1000000, 692)), Test: (100000, 692)
