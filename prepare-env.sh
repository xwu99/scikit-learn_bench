ENV_NAME=xgb15_conda_default
conda create -n $ENV_NAME python=3.9
source /home/xiaochang/miniconda3/bin/activate $ENV_NAME
conda install requests scikit-learn pandas xgboost tqdm openpyxl