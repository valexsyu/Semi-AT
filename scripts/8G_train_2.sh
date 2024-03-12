# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset samsum_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_folder /work/valex1377/semi_at_llama/llama_model/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_samsum \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss \
#          --num_epochs 3 \
#          --batch_size_training 3 \
#          --gradient_accumulation_steps 3 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#         #  --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \

# # dist_checkpoint_root_folder + dist_checkpoint_folder + model_name





# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset kd_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_folder /work/valex1377/semi_at_llama/llama_model/KD_ft_b3_WoKLDiv_WoSeqloss \
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



# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset kd_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_folder /work/valex1377/semi_at_llama/llama_model/KD_ft_b3_WoKLDiv_WiSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b-chat \
#          --num_epochs 3 \
#          --batch_size_training 3 \
#          --gradient_accumulation_steps 3 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#         #  --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \


# ## fsdp + lora
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset kd_dataset \
#          --use_peft --peft_method lora \
#          --enable_fsdp --pure_bf16 \
#          --output_dir /work/valex1377/semi_at_llama/llama_model/peft/KD_lora_b3_WoKLDiv_WoSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/7B-chat \
#          --num_epochs 1 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 3 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --sequence_label_ingnore \
# #          --target_kldiv_loss_enable \         

# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset samsum_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_folder KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_samsum_woself \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#          --num_epochs 3 \
#          --batch_size_training 3 \
#          --gradient_accumulation_steps 3 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#         #  --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \

# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset iwslt2017deennat_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_folder KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_iwslt17deen_woself \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#          --num_epochs 1 \
#          --batch_size_training 3 \
#          --gradient_accumulation_steps 1 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#         #  --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \

# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset openwebtext_kd_20k_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_folder /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_20k_ft_b3_WoKLDiv_WoSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b \
#          --num_epochs 3 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 9 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --sequence_label_ingnore \
#          --initialize_llamasemiat \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \

# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset openwebtext_kd_20k_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
#          --dist_checkpoint_folder KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b-chat \
#          --num_epochs 3 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 9 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#         #  --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \




# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset openwebtext_kd_80k_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
#          --dist_checkpoint_folder KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b \
#          --num_epochs 1 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 9 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --data_num 20000 \
#         #  --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \

# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset openwebtext_kd_80k_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
#          --dist_checkpoint_folder KD_opentext_gen_80k_80k_ft_b3_WoKLDiv_WiSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b \
#          --num_epochs 1 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 9 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --data_num 80000 \
#         #  --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \        



torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
         --dataset openwebtext_kd_80k_dataset \
         --enable_fsdp \
         --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
         --dist_checkpoint_folder KD_opentext_gen_80k_40k_ft_b3_WiKLDiv_WiSeqloss_WoChat \
         --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
         --num_epochs 1 \
         --batch_size_training 1 \
         --gradient_accumulation_steps 9 \
         --batching_strategy padding \
         --insert_token_num 1 \
         --target_kldiv_loss_enable \
         --data_num 40000 \
        #  --target_kldiv_loss_enable \
        #  --initialize_llamasemiat \
        #  --sequence_label_ingnore \

