ENV_NAME=xgb15_conda_default
conda create -n $ENV_NAME python=3.9
conda install requests scikit-learn pandas xgboost tqdm openpyxl daal4py
