ENV_NAME=xgb-predict-opt
conda create -n $ENV_NAME python=3.9
source /home/xiaochang/miniconda3/bin/activate $ENV_NAME
conda install requests scikit-learn pandas xgboost tqdm openpyxl daal4py
pip install hummingbird-ml[extra] apache-tvm onnxmltools onnxruntime