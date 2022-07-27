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

# Load and convert data
X_train, X_test, y_train, y_test = bench.load_data(params)
n_classes = len(np.unique(y_train))

print(f"Running with XGBoost {xgb.__version__}")
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
print(f"n_classes = {n_classes}\n")

# load saved model
print(f"Loading {params.model_file} ...")
booster = xgb.Booster()
booster.load_model(params.model_file)

classifier = xgb.XGBClassifier()
classifier.load_model("xgb-higgs1m-model.json")

# For multi-class
def convert_probs_to_classes(y_prob):
    return np.array([np.argmax(y_prob[i]) for i in range(y_prob.shape[0])])

def predict(dmatrix):  # type: ignore
    if dmatrix is None:
        dmatrix = xgb.DMatrix(X_test, y_test)
    prob_prediction = booster.predict(dmatrix)
    # output labels
    class_prediction = np.where(prob_prediction >= 0.5, 1.0, 0.0)
    # class_prediction = convert_probs_to_classes(prob_prediction)

    # xgb_errors_count = np.count_nonzero(predictions - y_test)
    return (prob_prediction, class_prediction)

def predict_onedal():
    import daal4py as d4p

    t0 = timeit.default_timer()

    # Conversion to daal4py
    daal_model = d4p.get_gbt_model_from_xgboost(booster)

    t1 = timeit.default_timer()
    print(f"--- Convert to oneDAL GBT model took {t1-t0:.3f} secs")

    t0 = timeit.default_timer()
    daal_predict_algo = d4p.gbt_classification_prediction(
        nClasses = n_classes,
        resultsToEvaluate="computeClassLabels|computeClassProbabilities",
        fptype='float'
    )
    daal_prediction = daal_predict_algo.compute(X_test, daal_model)
    t1 = timeit.default_timer()
    print(f"--- Predict with oneDAL GBT model took {t1-t0:.3f} secs")
    # daal_errors_count = np.count_nonzero(np.ravel(daal_prediction.prediction) - y_test)
    # print(daal_prediction.probabilities[0:20])
    # print(daal_prediction.probabilities[:,1])
    # print(daal_prediction.prediction)
    return (daal_prediction.probabilities[:,1], np.ravel(daal_prediction.prediction))

def predict_hummingbird():
    import hummingbird.ml

    t0 = timeit.default_timer()
    hummingbird_model = hummingbird.ml.convert(classifier, "tvm", X_test)
    t1 = timeit.default_timer()
    print(f"--- Convert to Hummingbird model took {t1-t0:.3f} secs")

    t0 = timeit.default_timer()
    class_prediction = hummingbird_model.predict(X_test)
    t1 = timeit.default_timer()
    print(f"--- Predict with Hummingbird model took {t1-t0:.3f} secs")

    return (None, class_prediction)


def run_predict_xgb():
    t0 = timeit.default_timer()
    (prob_prediction, class_prediction) = predict(None)
    t1 = timeit.default_timer()
    print(f"Predict using XGBoost Predictor took {t1-t0:.3f} secs")
    print(f"XGBoost prob_prediction results (first 10 rows): \n{prob_prediction[0:10]}\n")
    print(f"XGBoost prediction results (first 10 rows): \n{class_prediction[0:10]}\n")
    # print("XGBoost errors count:", xgb_errors_count)

def run_predict_onedal():
    t0 = timeit.default_timer()
    (prob_prediction, class_prediction) = predict_onedal()
    t1 = timeit.default_timer()
    print(f"Predict using oneDAL GBT Predictor took {t1-t0:.3f} secs")
    print(f"oneDAL prob_prediction results (first 10 rows): \n{prob_prediction[0:10]}\n")
    print(f"oneDAL prediction results (first 10 rows): \n{class_prediction[0:10]}\n")
    # print("\ndaal4py errors count:", daal_errors_count)
    # print("\nGround truth (first 10 rows):\n", y_test[0:10])

def run_predict_hummingbird():
    t0 = timeit.default_timer()
    (_, class_prediction) = predict_hummingbird()
    t1 = timeit.default_timer()

    print(f"Predict using Hummingbird Predictor took {t1-t0:.3f} secs")
    # print(f"Hummingbird prob_prediction results (first 10 rows): \n{prob_prediction[0:10]}\n")
    print(f"Hummingbird prediction results (first 10 rows): \n{class_prediction[0:10]}\n")

# print("")
# run_predict_xgb()
# print("")
# run_predict_onedal()
# print("")
run_predict_hummingbird()