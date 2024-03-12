# ## fsdp + ft
# python finetuning_kd.py \
#          --dataset kd_dataset \
#          --dist_checkpoint_folder /work/valex1377/semi_at_llama/llama_model/tests \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b-chat \
#          --num_epochs 1 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 3 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \                  


# ## fsdp + lora
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset kd_dataset \
#          --use_peft --peft_method lora \
#          --enable_fsdp --pure_bf16 \
#          --output_dir /work/valex1377/llama-nat/llama_model/peft/KD_lora_b3_WoKLDiv_WoSeqloss \
#          --model_name /work/valex1377/llama-nat/llama_model/7B-chat \
#          --num_epochs 1 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 3 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --sequence_label_ingnore \
# #          --target_kldiv_loss_enable \        



# ## fsdp + ft
# python finetuning_kd.py \
#          --dataset openwebtext_kd_20k_dataset \
#          --dist_checkpoint_folder /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_20k_ft_b3_WoKLDiv_WoSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b-chat \
#          --num_epochs 3 \
#          --batch_size_training 3 \
#          --gradient_accumulation_steps 3 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \


## fsdp + ft
python finetuning_kd.py \
         --dataset openwebtext_kd_80k_dataset \
         --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
         --dist_checkpoint_folder test1123 \
         --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
         --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b-chat \
         --num_epochs 1 \
         --batch_size_training 1 \
         --gradient_accumulation_steps 3 \
         --batching_strategy padding \
         --insert_token_num 1 \
         --sequence_label_ingnore \
         --data_num 30000 \
        #  --add_diff_para \
        #  --chat_model_path /work/valex1377/semi_at_llama/llama_model/2epoch/KD_opentext_gen_20k_ft_b3_WoKLDiv_WoSeqloss \
        #  --no_chat_model_path /work/valex1377/semi_at_llama/llama_model/2epoch/KD_opentext_gen_20k_ft_b3_WoKLDiv_WoSeqloss_wochat \

        #  --target_kldiv_loss_enable \   
        #  --initialize_llamasemiat \       




 