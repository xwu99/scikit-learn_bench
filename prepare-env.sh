ENV_NAME=xgb15-conda-default
conda create -y -n $ENV_NAME python=3.9
source /home/xiaochang/miniconda3/bin/activate $ENV_NAME
conda install -y numpy pandas scikit-learn requests xgboost tqdm openpyxl