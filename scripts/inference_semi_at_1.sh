# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/model_outputs/LlamaSemiATForCausalLM \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \


# --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
# --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b1_WoKLDiv_WiSeqloss \

# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/7B-chat \
#          --insert_token_num 1 \
#          --insert_token_id 0 \
#          --use_cache \
         

# --use_peft --peft_method lora \
# --output_dir /work/valex1377/llama/PEFT/chat_nat_lora_batch_padloss_e7_up2-last \
# --more_step 0 \
# --result_file_prefix _special_ufixed-prompt_up1

# python  inference_semi_at.py \
#          --dataset alpaca_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \

# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --insert_waiting \
#          --result_folder add_diff_para_insert_waiting
#         #  --chat_model_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#         #  --no_chat_model_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \         
#         #  --insert_waiting \

#         # ---------dataset-------------
#         # iwslt2017deennat_dataset
#         # alpaca_dataset
#         # samsum_dataset
#         # openwebtext_kd_20k_dataset

#         ##---------model_name-----------
#         # LlamaSemiATForCausalLM
#         # LlamaSemiATForCausalLM_wochat

#         # ---------self distill---------
#         # KD_ft_b3_WoKLDiv_WiSeqloss
#         # KD_opentext_gen_20k_ft_b3_WoKLDiv_WoSeqloss_wochat_2e
#         # KD_opentext_gen_20k_ft_b3_WoKLDiv_WoSeqloss_2e
#         # KD_opentext_gen_20k_ft_b3_WoKLDiv_WoSeqloss_wochat_1e
#         # KD_opentext_gen_20k_ft_b3_WoKLDiv_WoSeqloss_1e        
#         #   =====iwslt17deen=====
#         # KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_iwslt17deen
#         #   =====samsum=====
#         # KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_samsum_woself
#         # KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_samsum
#         #   =====alpaca======
#         # KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset
#         # KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset_woself
#         #-----------delta model for chat----------
#         #  --add_diff_para \
#         #  --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#         #  --chat_model_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#         #  --no_chat_model_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \        

#         #  --result_folder iwslt2017deen \

# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --insert_waiting \
#          --result_folder add_diff_para_insert_waiting



# echo "3"
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting

# echo "4"
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting


# echo "5"
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_wochat_2e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --result_folder add_diff_para

# echo "6"
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_wochat_2e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting 

               

# echo "=============7==================="
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --result_folder add_diff_para

# echo "=============8============="
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting               
         

# echo "=============9============="
# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting           

# echo "================10==================="
# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --result_folder add_diff_para        

         
# echo "================11================"
# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_wochat_2e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting 


# echo "================12================"
# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_wochat_2e \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --result_folder add_diff_para         



# echo "=============1==================="
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --result_folder add_diff_para

# echo "=============2==================="
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --result_folder insert_waiting_add_diff_para

# echo "=============3==================="
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting

# echo "=============4==================="
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --result_folder no_insert_waiting         




# echo "=============5==================="
# python  inference_semi_at.py \
#          --dataset mt_bench_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b \
#          --use_cache \
#          --result_folder no_insert_waiting  \
#          --max_new_tokens 1000

# echo "=============6==================="
# python  inference_semi_at.py \
#          --dataset mt_bench_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b \
#          --use_cache \
#          --insert_token_num 0 \
#          --result_folder at  \
#          --max_new_tokens 1000




# echo "=============1==================="
# python  inference_semi_at.py \
#          --dataset mt_bench_dataset \
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
#          --dataset mt_bench_dataset \
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
#          --dataset mt_bench_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting \
#          --max_new_tokens 1000

# echo "=============4==================="
# python  inference_semi_at.py \
#          --dataset mt_bench_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --result_folder no_insert_waiting  \
#          --max_new_tokens 1000




# echo "=============1==================="
# python  inference_semi_at.py \
#          --dataset mt_bench_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_80k_ft_b3_WoKLDiv_WiSeqloss_wochat \
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
#          --dataset mt_bench_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_80k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --add_diff_para \
#          --delta_para_path /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt \
#          --result_folder add_diff_para \
#          --max_new_tokens 1000


# echo "=============3==================="
# python  inference_semi_at.py \
#          --dataset mt_bench_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_80k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting \
#          --max_new_tokens 1000

# echo "=============4==================="
# python  inference_semi_at.py \
#          --dataset mt_bench_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_80k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --result_folder no_insert_waiting  \
#          --max_new_tokens 1000



# echo "=============3==================="
# python  inference_semi_at.py \
#          --dataset iwslt2017deennat_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting \
#          --max_new_tokens 1000




# echo "=============3==================="
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --insert_waiting \
#          --result_folder insert_waiting

# echo "=============4==================="
# python  inference_semi_at.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat \
#          --insert_token_num 1 \
#          --insert_token_id 32000 \
#          --use_cache \
#          --result_folder no_insert_waiting    



echo "=============3==================="
python  inference_semi_at.py \
         --dataset samsum_dataset \
         --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat_WoNoise \
         --insert_token_num 1 \
         --insert_token_id 32000 \
         --use_cache \
         --insert_waiting \
         --result_folder insert_waiting

echo "=============4==================="
python  inference_semi_at.py \
         --dataset samsum_dataset \
         --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat_WoNoise \
         --insert_token_num 1 \
         --insert_token_id 32000 \
         --use_cache \
         --result_folder no_insert_waiting   
