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



## fsdp + ft
torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
         --dataset alpaca_dataset \
         --enable_fsdp \
         --dist_checkpoint_folder KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset \
         --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss \
         --num_epochs 1 \
         --batch_size_training 3 \
         --gradient_accumulation_steps 1 \
         --batching_strategy padding \
         --insert_token_num 1 \
        #  --sequence_label_ingnore \
        #  --target_kldiv_loss_enable \   
        #  --initialize_llamasemiat \

## fsdp + ft
torchrun --nnodes 1 --nproc_per_node 8 finetuning_kd.py \
         --dataset alpaca_dataset \
         --enable_fsdp \
         --dist_checkpoint_folder KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset_woself \
         --model_name /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM \
         --num_epochs 1 \
         --batch_size_training 3 \
         --gradient_accumulation_steps 1 \
         --batching_strategy padding \
         --insert_token_num 1 \
        #  --sequence_label_ingnore \
        #  --target_kldiv_loss_enable \   
        #  --initialize_llamasemiat \        