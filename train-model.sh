#!/usr/bin/env bash

source /home/xiaochang/miniconda3/bin/activate xgb15-conda-default

export PYTHONPATH=${PWD}

# python xgboost_bench/gbt_train.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --learning-rate 0.03 --max-depth 6 --n-estimators 1000 --objective reg:squarederror --device none --file-X-train data/abalone_x_train.npy --file-y-train data/abalone_y_train.npy --file-X-test data/abalone_x_test.npy --file-y-test data/abalone_y_test.npy --dataset-name abalone

# python xgboost_bench/gbt_train.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --device none --file-X-train data/airline-ohe_x_train.npy --file-y-train data/airline-ohe_y_train.npy --file-X-test data/airline-ohe_x_test.npy --file-y-test data/airline-ohe_y_test.npy --dataset-name airline-ohe

# -- Selected --

python xgboost_bench/gbt_train.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --n-estimators 10000 --objective binary:logistic --max-depth 1 --subsample 0.5 --eta 0.1 --colsample-bytree 0.05 --device none --file-X-train data/santander_x_train.npy --file-y-train data/santander_y_train.npy --file-X-test data/santander_x_test.npy --file-y-test data/santander_y_test.npy --dataset-name santander

python xgboost_bench/gbt_train.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy --dataset-name higgs1m

python xgboost_bench/gbt_train.py --arch mlp-sdp-spr-7639 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --max-bin 256 --learning-rate 0.3 --subsample 1 --reg-lambda 2 --min-child-weight 1 --min-split-loss 0.1 --max-depth 8 --n-estimators 200 --objective multi:softprob --device none --file-X-train data/mlsr_x_train.npy --file-y-train data/mlsr_y_train.npy --dataset-name mlsr