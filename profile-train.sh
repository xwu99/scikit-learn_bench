#!/usr/bin/env bash

source /home/xiaochang/miniconda3/bin/activate xgb15_conda_default
# source /home/xiaochang/miniconda3/bin/activate xgb-intel
# source /home/xiaochang/miniconda3/bin/activate xgb-kirill-icx

# higgs1m NUMA-bind
# export OMP_PLACES="{0:72}"
# export OMP_PLACES="{0:72}"
# export OMP_PROC_BIND=true
# numactl -i all -- python xgboost_bench/gbt_train.py --arch sr271 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads 72 --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy --dataset-name higgs1m
# numactl -N 0 -m 0 -- python xgboost_bench/gbt_train.py --arch sr271 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads 36 --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy --dataset-name higgs1m

# higgs1m
python xgboost_bench/gbt_train.py --arch sr271 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads 72 --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy --dataset-name higgs1m

# mslr_web30k
# python xgboost_bench/gbt_train.py --arch sr271 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads 80 --max-bin 256 --learning-rate 0.3 --subsample 1 --reg-lambda 2 --min-child-weight 1 --min-split-loss 0.1 --max-depth 8 --n-estimators 200 --objective multi:softprob --single-precision-histogram --device none --file-X-train data/mlsr_x_train.npy --file-y-train data/mlsr_y_train.npy --dataset-name mlsr

# airline-ohe
# python xgboost_bench/gbt_train.py --arch sr271 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads 72 --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --device none --file-X-train data/airline-ohe_x_train.npy --file-y-train data/airline-ohe_y_train.npy --file-X-test data/airline-ohe_x_test.npy --file-y-test data/airline-ohe_y_test.npy --dataset-name airline-ohe