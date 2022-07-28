#!/usr/bin/env bash

# source /home/xiaochang/miniconda3/bin/activate xgb15-conda-default
# source /home/xiaochang/miniconda3/bin/activate xgb12
source /home/xiaochang/miniconda3/bin/activate xgb-predict-opt

export PYTHONPATH=${PWD}
NUM_THREADS=$(nproc)

# higgs1m shape: Train: (1000000, 28), Test: (500000, 28)
# python -W ignore xgboost_bench/gbt_predict.py \
#   --model-file xgb-higgs1m-model.json \
#   --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy \
#   --dataset-name higgs1m

# higgs1m w/ scaled test dataset shape: Train: (1000000, 28), Test: (50000000, 28)
python -W ignore xgboost_bench/gbt_predict.py \
  --model-file xgb-higgs1m-model.json \
  --num-threads $NUM_THREADS \
  --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_100x_tile_x_test.npy --file-y-test data/higgs1m_100x_tile_y_test.npy \
  --dataset-name higgs1m

# airline-ohe shape: Train: (1000000, 692)), Test: (100000, 692)
# python xgboost_bench/gbt_predict.py \
#   --model-file xgb-airline-ohe-model.json \
#   --file-X-train data/airline-ohe_x_train.npy --file-y-train data/airline-ohe_y_train.npy --file-X-test data/airline-ohe_x_test.npy --file-y-test data/airline-ohe_y_test.npy \
#   --dataset-name airline-ohe