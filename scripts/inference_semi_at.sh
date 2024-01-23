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

python  inference_semi_at.py \
         --dataset alpaca_dataset \
         --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
         --insert_token_num 1 \
         --insert_token_id 32000 \
         --use_cache \