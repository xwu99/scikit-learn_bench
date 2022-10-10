ENV_NAME=xgb15-conda-default
conda create -y -n $ENV_NAME python=3.9
source /home/xiaochang/miniconda3/bin/activate $ENV_NAME
conda install -y numpy pandas scikit-learn=1.0 scikit-learn-intelex requests xgboost tqdm openpyxl
