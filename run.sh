unzip final.zip
mv final data
code_path=./
cd code
python $code_path"_1_preprocess_data.py"
python $code_path"_2_1_gen_user_click_features.py"
python $code_path"_2_2_gen_app_install_features.py"
python $code_path"_2_3_gen_global_sum_counts.py"
python $code_path"_2_4_gen_tricks.py"
python $code_path"_2_5_gen_smooth_cvr.py"
python $code_path"_2_6_gen_ID_click_vectors.py"
python $code_path"_2_7_gen_trick_final.py"
python $code_path"_3_0_gen_final_data.py"
echo "finished gen data!!!"