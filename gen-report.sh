# FILE1=../spr_pip_xgb15.json
# FILE2=../icx_pip_xgb15.json
# OUTPUT=spr-vs-icx-pip-xgb15.xlsx
# python report_generator.py --result-files $FILE1,$FILE2 --report-file $OUTPUT

# FILE1=../spr_pip_xgb15_noht.json
# FILE2=../icx_pip_xgb15_noht.json
# OUTPUT=spr-vs-icx-pip-xgb15-noht.xlsx
# python report_generator.py --result-files $FILE1,$FILE2 --report-file $OUTPUT

# FILE1=../spr_xgb15_conda_default_noht.json
# FILE2=../icx_xgb15_conda_default_noht.json
# OUTPUT=spr-vs-icx-xgb15-conda_default-noht.xlsx
# python report_generator.py --result-files $FILE1,$FILE2 --report-file $OUTPUT
# FILES=../0_results/xgb_epsilon_icx_xgb-intel/xgb_epsilon_icx_xgb-intel_run1.json,\
# ../0_results/xgb_epsilon_icx_xgb-intel/xgb_epsilon_icx_xgb-intel_run2.json,\
# ../0_results/xgb_epsilon_icx_xgb-intel/xgb_epsilon_icx_xgb-intel_run3.json,\
# ../0_results/xgb_epsilon_icx_icx_xgb15_conda_default/xgb_epsilon_icx_icx_xgb15_conda_default_run1.json,\
# ../0_results/xgb_epsilon_icx_icx_xgb15_conda_default/xgb_epsilon_icx_icx_xgb15_conda_default_run2.json,\
# ../0_results/xgb_epsilon_icx_icx_xgb15_conda_default/xgb_epsilon_icx_icx_xgb15_conda_default_run3.json

cd report_generator

FILES=\
../0_results/xgb_cpu_additional_config_icx_xgb-intel/xgb_cpu_additional_config_icx_xgb-intel_run1.json,\
../0_results/xgb_cpu_additional_config_icx_xgb-intel/xgb_cpu_additional_config_icx_xgb-intel_run2.json,\
../0_results/xgb_cpu_additional_config_icx_xgb-intel/xgb_cpu_additional_config_icx_xgb-intel_run3.json,\
../0_results/xgb_cpu_additional_config_icx_xgb15_conda_default/xgb_cpu_additional_config_icx_xgb15_conda_default_run1.json,\
../0_results/xgb_cpu_additional_config_icx_xgb15_conda_default/xgb_cpu_additional_config_icx_xgb15_conda_default_run2.json,\
../0_results/xgb_cpu_additional_config_icx_xgb15_conda_default/xgb_cpu_additional_config_icx_xgb15_conda_default_run3.json

OUTPUT=../0_results/xgb_cpu_additional_icx_intel_vs_vanilla.xlsx
GEN_CONFIG=custom_report_gen_config.json

python report_generator.py --result-files $FILES --report-file $OUTPUT --generation-config $GEN_CONFIG

# cd report_generator
# BASE_NAME=xgb_pandas_F_spr
# for run_no in {1..3}
# do
# RESULT_FILE=../0_results/${BASE_NAME}_run${run_no}.json
# REPORT_FILE=../0_results/${BASE_NAME}_run${run_no}.xlsx
# python report_generator.py --result-files $RESULT_FILE --report-file $REPORT_FILE
# done
