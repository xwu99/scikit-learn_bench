#!/usr/bin/env bash

# source /home/xiaochang/miniconda3/bin/activate xgb15_conda_default
source /home/xiaochang/miniconda3/bin/activate xgb-daal4py

python xgboost_bench/gbt_predict.py \
  --model-file xgb-higgs1m.model \
  --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy \
  --dataset-name higgs1m

# python xgboost_bench/gbt_predict.py \
#   --model-file xgb-higgs1m.model \
#   --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_100x_tile_x_test.npy --file-y-test data/higgs1m_100x_tile_y_test.npy \
#   --dataset-name higgs1m

# python xgboost_bench/gbt_predict.py \
#   --model-file xgb-airline-ohe.model \
#   --file-X-train data/airline-ohe_x_train.npy --file-y-train data/airline-ohe_y_train.npy --file-X-test data/airline-ohe_x_test.npy --file-y-test data/airline-ohe_y_test.npy \
#   --dataset-name airline-ohe