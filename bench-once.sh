source /home/xiaochang/miniconda3/bin/activate xgb15_conda_default

CONFIG_FILES=configs_profile/xgb_airline_icx.json

BASE_NAME=$(basename -s .json $CONFIG_FILES)_${CONDA_DEFAULT_ENV}
TIMESTAMP=$(date +%m%d_%H%M%S)
mkdir -p 0_results/${BASE_NAME}

OUTPUT_FILE=0_results/${BASE_NAME}/${BASE_NAME}_${TIMESTAMP}.json
LOG_FILE=0_results/${BASE_NAME}/${BASE_NAME}_${TIMESTAMP}.log

python runner.py --configs $CONFIG_FILES --output-file $OUTPUT_FILE | tee $LOG_FILE
