python  llama_inference.py \
         --dataset natsamsum_dataset \
         --model_name /work/valex1377/llama/models_hf/7B-chat \
         --use_peft --peft_method lora \
         --output_dir /work/valex1377/llama/PEFT/chat_nat_lora_batch_padloss_e7_up2-last \
         --num_epochs 1 \
         --upsampling_rate 1 \
         --special_decode \
         --use_cache \
         --more_step 0 \
         --result_file_prefix _special_ufixed-prompt_up1
