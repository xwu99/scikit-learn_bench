#!/usr/bin/env bash

source /home/xiaochang/miniconda3/bin/activate xgb15-conda-default

export PYTHONPATH=${PWD}
NUM_THREADS=72
MAX_DEPTH=8
MAX_LEAVES=256
NUM_TREES=1000
MAX_BIN=256

# airline-ohe numpy C
# python xgboost_bench/gbt_train.py --arch sr271 --data-format numpy --data-order C --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --device none --file-X-train data/airline-ohe_x_train.npy --file-y-train data/airline-ohe_y_train.npy --file-X-test data/airline-ohe_x_test.npy --file-y-test data/airline-ohe_y_test.npy --dataset-name airline-ohe

# airline-ohe pandas F
# python xgboost_bench/gbt_train.py --arch sr271 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --device none --file-X-train data/airline-ohe_x_train.npy --file-y-train data/airline-ohe_y_train.npy --file-X-test data/airline-ohe_x_test.npy --file-y-test data/airline-ohe_y_test.npy --dataset-name airline-ohe

# higgs1m numpy C
# python xgboost_bench/gbt_train.py --arch sr271 --data-format numpy --data-order C --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy --dataset-name higgs1m

# higgs1m pandas F
# python xgboost_bench/gbt_train.py --arch sr271 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --reg-alpha 0.9 --max-bin 256 --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth 8 --max-leaves 256 --n-estimators 1000 --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy --dataset-name higgs1m

# mslr_web30k numpy C
# python xgboost_bench/gbt_train.py --arch sr271 --data-format numpy --data-order C --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --max-bin 256 --learning-rate 0.3 --subsample 1 --reg-lambda 2 --min-child-weight 1 --min-split-loss 0.1 --max-depth 8 --n-estimators 200 --objective multi:softprob --single-precision-histogram --device none --file-X-train data/mlsr_x_train.npy --file-y-train data/mlsr_y_train.npy --dataset-name mlsr

# mslr_web30k pandas F
# python xgboost_bench/gbt_train.py --arch sr271 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --max-bin 256 --learning-rate 0.3 --subsample 1 --reg-lambda 2 --min-child-weight 1 --min-split-loss 0.1 --max-depth 8 --n-estimators 200 --objective multi:softprob --single-precision-histogram --device none --file-X-train data/mlsr_x_train.npy --file-y-train data/mlsr_y_train.npy --dataset-name mlsr

# higgs1m parameter tuning
python xgboost_bench/gbt_train.py --arch sr271 --data-format pandas --data-order F --dtype float32 --tree-method hist --count-dmatrix --num-threads $NUM_THREADS --reg-alpha 0.9 --max-bin $MAX_BIN --scale-pos-weight 2 --learning-rate 0.1 --subsample 1 --reg-lambda 1 --min-child-weight 0 --max-depth $MAX_DEPTH --max-leaves $MAX_LEAVES --n-estimators $NUM_TREES --objective binary:logistic --enable-experimental-json-serialization False --inplace-predict --device none --file-X-train data/higgs1m_x_train.npy --file-y-train data/higgs1m_y_train.npy --file-X-test data/higgs1m_x_test.npy --file-y-test data/higgs1m_y_test.npy --dataset-name higgs1m | tee 0_results/higgs1m-pandas-F-${NUM_THREADS}threads-${NUM_TREES}trees-${MAX_DEPTH}depth-${MAX_LEAVES}leaves-${MAX_BIN}bins.log