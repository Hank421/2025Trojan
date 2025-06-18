python trojan_xgboost.py
python predict_final_results_v2.py
python score.py

# for i in $(seq 0.4 0.05 0.6)
# do
#     for j in $(seq 0.65 0.05 0.95)
#     do
#         python predict_final_results_v2.py --prob1 $i --prob2 $j
#         echo "Running with prob1=$i and prob2=$j" >> exp_dual_score.txt
#         python score.py >> exp_dual_score.txt
#     done
# done
