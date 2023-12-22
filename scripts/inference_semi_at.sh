python  inference_semi_at.py \
         --dataset natsamsum_dataset \
         --model_name /home/valexsyu/Documents/NAT/Semi-AT/llama_model/models_hf/LlamaSemiATForCausalLM \
         --num_epochs 1 \
         --insert_token_num 1 \
         --use_cache \

         

# --use_peft --peft_method lora \
# --output_dir /work/valex1377/llama/PEFT/chat_nat_lora_batch_padloss_e7_up2-last \
# --more_step 0 \
# --result_file_prefix _special_ufixed-prompt_up1