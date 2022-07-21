TIMESTAMP=$(date +%m%d_%H%M%S)
advisor --collect=survey --project-dir=./advisor_results_${TIMESTAMP} -- ./profile-train.sh
advisor --report=survey --project-dir=./advisor_results_${TIMESTAMP} --report-output=./advisor-report.txt
# advisor --collect=roofline --project-dir=./advisor_results -- ./profile-train.sh
# advisor --collect=map --project-dir=./advisor_results -- ./profile-train.sh
