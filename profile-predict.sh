#!/usr/bin/env bash

# source /home/xiaochang/miniconda3/bin/activate xgb15_conda_default
source /home/xiaochang/miniconda3/bin/activate xgb-daal4py

python xgboost_bench/gbt_predict.py \
  --model-file xgb-higgs1m.model \
  --file-X-train data/higgs1m_x_train.npy \
  --file-y-train data/higgs1m_y_train.npy \
  --file-X-test data/higgs_x_train.npy \
  --file-y-test data/higgs_y_train.npy \
  --dataset-name higgs1m