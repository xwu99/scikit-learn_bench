# source /home/xiaochang/miniconda3/bin/activate xgb15_conda_default

# CONFIG_FILES=configs_benchmark/xgb_pandas_F_icx.json
# CONFIG_FILES=configs_benchmark/xgb_plasticc_icx.json
# CONFIG_FILES=configs_benchmark/xgb_letters_icx.json
# CONFIG_FILES=configs_benchmark/xgb_cpu_additional_config_icx.json
CONFIG_FILES=configs_benchmark/xgb_higgs1m_icx.json
# CONFIG_FILES=configs_benchmark/xgb_epsilon_icx.json
BASE_NAME=$(basename -s .json $CONFIG_FILES)_${CONDA_DEFAULT_ENV}
mkdir -p 0_results/${BASE_NAME}
cp ${CONFIG_FILES} 0_results/${BASE_NAME}
for run_no in {1..3}
do
OUTPUT_FILE=0_results/${BASE_NAME}/${BASE_NAME}_run${run_no}.json
LOG_FILE=0_results/${BASE_NAME}/${BASE_NAME}_run${run_no}.log
# python runner.py --configs $CONFIG_FILES --output-file $OUTPUT_FILE --report | tee $LOG_FILE
python runner.py --configs $CONFIG_FILES --output-file $OUTPUT_FILE | tee $LOG_FILE
done
