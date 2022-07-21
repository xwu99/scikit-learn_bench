# ===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import argparse

import bench
import numpy as np
import xgboost as xgb

import timeit


def convert_probs_to_classes(y_prob):
    return np.array([np.argmax(y_prob[i]) for i in range(y_prob.shape[0])])


def convert_xgb_predictions(y_pred, objective):
    if objective == 'multi:softprob':
        y_pred = convert_probs_to_classes(y_pred)
    elif objective == 'binary:logistic':
        y_pred = y_pred.astype(np.int32)
    return y_pred


parser = argparse.ArgumentParser(description='xgboost gradient boosted trees benchmark')

parser.add_argument('--model-file', type=str, required=True,
                    help='The XGBoost model file for prediciton')

params = bench.parse_args(parser)

# Default seed
if params.seed == 12345:
    params.seed = 0

# Load and convert data
X_train, X_test, y_train, y_test = bench.load_data(params)

# xgb_params = {
#     'booster': 'gbtree',
#     'verbosity': 0,
#     'learning_rate': params.learning_rate,
#     'min_split_loss': params.min_split_loss,
#     'max_depth': params.max_depth,
#     'min_child_weight': params.min_child_weight,
#     'max_delta_step': params.max_delta_step,
#     'subsample': params.subsample,
#     'sampling_method': 'uniform',
#     'colsample_bytree': params.colsample_bytree,
#     'colsample_bylevel': 1,
#     'colsample_bynode': 1,
#     'reg_lambda': params.reg_lambda,
#     'reg_alpha': params.reg_alpha,
#     'tree_method': params.tree_method,
#     'scale_pos_weight': params.scale_pos_weight,
#     'grow_policy': params.grow_policy,
#     'max_leaves': params.max_leaves,
#     'max_bin': params.max_bin,
#     'objective': params.objective,
#     'seed': params.seed,
#     'single_precision_histogram': params.single_precision_histogram,
#     'enable_experimental_json_serialization':
#         params.enable_experimental_json_serialization
# }

# if params.threads != -1:
#     xgb_params.update({'nthread': params.threads})

# if params.objective.startswith('reg'):
#     task = 'regression'
#     metric_name, metric_func = 'rmse', bench.rmse_score
# else:
#     task = 'classification'
#     metric_name = 'accuracy'
#     metric_func = bench.accuracy_score
#     if 'cudf' in str(type(y_train)):
#         params.n_classes = y_train[y_train.columns[0]].nunique()
#     else:
#         params.n_classes = len(np.unique(y_train))

#     # Covtype has one class more than there is in train
#     if params.dataset_name == 'covtype':
#         params.n_classes += 1

#     if params.n_classes > 2:
#         xgb_params['num_class'] = params.n_classes

# dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)


# def fit(dmatrix):
#     if dmatrix is None:
#         dmatrix = xgb.DMatrix(X_train, y_train)
#     return xgb.train(xgb_params, dmatrix, params.n_estimators)


# fit(None)

# load saved model
print(f"Loading {params.model_file} ...")
booster = xgb.Booster()
booster.load_model(params.model_file)

def predict(dmatrix):  # type: ignore
    if dmatrix is None:
        dmatrix = xgb.DMatrix(X_test, y_test)
    xgb_prediction = booster.predict(dmatrix)
    # predictions = np.array([1. if (value >= 0.5) else 0. for value in xgb_prediction])

    # xgb_errors_count = np.count_nonzero(predictions - y_test)
    # return (predictions, None)
    return (xgb_prediction, None)

def predict_onedal():
    import daal4py as d4p

    t0 = timeit.default_timer()

    # Conversion to daal4py
    daal_model = d4p.get_gbt_model_from_xgboost(booster)

    t1 = timeit.default_timer()
    print("--- Convert to oneDAL GBT model took " + str(t1 - t0) + " secs")

    t0 = timeit.default_timer()
    daal_predict_algo = d4p.gbt_classification_prediction(
        nClasses=2,
        resultsToEvaluate="computeClassLabels",
        fptype='float'
    )
    daal_prediction = daal_predict_algo.compute(X_test, daal_model)
    t1 = timeit.default_timer()
    print("--- Predict with oneDAL GBT model took " + str(t1 - t0) + " secs")
    # daal_errors_count = np.count_nonzero(np.ravel(daal_prediction.prediction) - y_test)
    return (np.ravel(daal_prediction.prediction), None)


t0 = timeit.default_timer()
(xgb_prediction, xgb_errors_count) = predict(None)
t1 = timeit.default_timer()
print("Predict using XGBoost Predictor took " + str(t1 - t0) + " secs")

print(f"\nXGBoost prediction results (first 10 rows): \n{xgb_prediction[0:10]}\n")
# print("XGBoost errors count:", xgb_errors_count)

t0 = timeit.default_timer()
(daal_prediction, _) = predict_onedal()
t1 = timeit.default_timer()
print("Predict using oneDAL GBT Predictor took " + str(t1 - t0) + " secs")

print(f"\ndaal4py prediction results (first 10 rows): \n{daal_prediction[0:10]}\n")
# print("\ndaal4py errors count:", daal_errors_count)

# print("\nGround truth (first 10 rows):\n", y_test[0:10])