





# echo "=============1==================="
# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --result_folder insert_waiting_add_diff_para \
#          --max_new_tokens 1000

# echo "=============2==================="
# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --result_folder add_diff_para \
#          --max_new_tokens 1000


# echo "=============3==================="
# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting \
#          --max_new_tokens 1000

# echo "=============4==================="
# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --result_folder no_insert_waiting  \
#          --max_new_tokens 1000







