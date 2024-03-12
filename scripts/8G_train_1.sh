# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset iwslt2017deennat_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_folder KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_iwslt17deen \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss \
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
#          --dataset alpaca_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_folder KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss \
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
#          --dataset alpaca_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_folder KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset_woself \
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
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b-chat \
#          --num_epochs 3 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 9 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \

# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset openwebtext_kd_80k_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
#          --dist_checkpoint_folder KD_opentext_gen_80k_40k_ft_b3_WoKLDiv_WiSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b \
#          --num_epochs 1 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 9 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --data_num 40000 \
#         #  --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \

# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset openwebtext_kd_80k_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
#          --dist_checkpoint_folder KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
#          --llama_model /work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b \
#          --num_epochs 1 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 9 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --data_num 60000 \
#         #  --sequence_label_ingnore \
#         #  --target_kldiv_loss_enable \   
#         #  --initialize_llamasemiat \   







# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset openwebtext_kd_80k_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
#          --dist_checkpoint_folder KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
#          --num_epochs 1 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 9 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --target_kldiv_loss_enable \
#          --data_num 20000 
#         #  --noise_rate 0
#         #  --target_kldiv_loss_enable \
#         #  --initialize_llamasemiat \
#         #  --sequence_label_ingnore \





# ## fsdp + ft
# torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
#          --dataset openwebtext_kd_80k_dataset \
#          --enable_fsdp \
#          --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
#          --dist_checkpoint_folder KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat_WoNoise \
#          --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
#          --num_epochs 1 \
#          --batch_size_training 1 \
#          --gradient_accumulation_steps 9 \
#          --batching_strategy padding \
#          --insert_token_num 1 \
#          --target_kldiv_loss_enable \
#          --data_num 20000 \
#          --noise_rate 0
#         #  --target_kldiv_loss_enable \
#         #  --initialize_llamasemiat \
#         #  --sequence_label_ingnore \


torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
         --dataset openwebtext_kd_80k_dataset \
         --enable_fsdp \
         --dist_checkpoint_root_folder /work/valex1377/semi_at_llama/llama_model \
         --dist_checkpoint_folder KD_opentext_gen_80k_40k_ft_b3_WiKLDiv_WiSeqloss_WoChat_WoNoise \
         --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat \
         --num_epochs 1 \
         --batch_size_training 1 \
         --gradient_accumulation_steps 9 \
         --batching_strategy padding \
         --insert_token_num 1 \
         --target_kldiv_loss_enable \
         --data_num 40000 \
         --noise_rate 0
        #  --target_kldiv_loss_enable \
        #  --initialize_llamasemiat \
        #  --sequence_label_ingnore \

