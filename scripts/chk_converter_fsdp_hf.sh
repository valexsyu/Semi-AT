# python inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/llama-recipes/model_checkpoints/fine-tuned-/work/valex1377/llama/models_hf/7B \
#        /work/valex1377/llama/models_hf/finetune-7B-nonlora-1epoch \
#        /work/valex1377/llama/models_hf/7B


#     fsdp_checkpoint_path="", # Path to FSDP Sharded model checkpoints
#     consolidated_model_path="", # Path to save the HF converted model checkpoints
#     HF_model_path_or_name="" # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
# python inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/llama-recipes/model_checkpoints/ft_e7_up1-/work/valex1377/llama/models_hf/7B \
#        /work/valex1377/llama/models_hf/finetune-7B-ft-1epoch \
#        /work/valex1377/llama/models_hf/7B
# python inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/llama-recipes/model_checkpoints/ft_e7_up1-last-/work/valex1377/llama/models_hf/7B \
#        /work/valex1377/llama/models_hf/finetune-7B-ft-7epoch \
#        /work/valex1377/llama/models_hf/7B
 

# python inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/llama-recipes/model_checkpoints/chat_at_ft_batch_padloss_e7/work/valex1377/llama/models_hf/7B-chat \
#        /work/valex1377/llama/models_hf/chat_at_ft_batch_padloss_e7 \
#        /work/valex1377/llama/models_hf/7B-chat        

# python inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/llama-recipes/model_checkpoints/chat_at_ft_batch_padloss_e1/work/valex1377/llama/models_hf/7B-chat \
#        /work/valex1377/llama/models_hf/chat_at_ft_batch_padloss_e1 \
#        /work/valex1377/llama/models_hf/7B-chat          


# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/model_outputs/KD_ft_b1_WoKLDiv_WiSeqloss \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b1_WoKLDiv_WiSeqloss \
#        /work/valex1377/semi_at_llama/model_outputs/LlamaSemiATForCausalLM  

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/model_outputs/KD_ft_b1_WoKLDiv_WoSeqloss \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b1_WoKLDiv_WoSeqloss \
#        /work/valex1377/semi_at_llama/model_outputs/LlamaSemiATForCausalLM

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/model_outputs/KD_ft_b1_WoKLDiv_WoSeqloss \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b1_WoKLDiv_WoSeqloss \
#        /work/valex1377/semi_at_llama/model_outputs/LlamaSemiATForCausalLM       

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/model_outputs/KD_ft_b1_WoKLDiv_WiSeqloss \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b1_WoKLDiv_WiSeqloss \
#        /work/valex1377/semi_at_llama/model_outputs/LlamaSemiATForCausalLM    

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_iwslt17deen \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_iwslt17deen \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss         

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_samsum \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_samsum \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss         

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss        

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset_woself \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_alpaca_dataset_woself \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM        

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_samsum_woself \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_samsum_woself \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM        



# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_iwslt17deen_woself \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_ft_b3_WoKLDiv_WiSeqloss_ft_b3_iwslt17deen_woself \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM        


# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_wochat_2e \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_wochat_2e \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat


# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM


# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_nochat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_nochat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_80k_40k_ft_b3_WoKLDiv_WiSeqloss_nochat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_40k_ft_b3_WoKLDiv_WiSeqloss_nochat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat       

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat


# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_80k_80k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_80k_ft_b3_WoKLDiv_WiSeqloss_wochat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat


       

# python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
#        /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat \
#        /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat



python /work/valex1377/semi_at_llama/llama_recipes/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
       /work/valex1377/semi_at_llama/llama_model/KD_opentext_gen_80k_40k_ft_b3_WiKLDiv_WiSeqloss_WoChat_WoNoise \
       /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_40k_ft_b3_WiKLDiv_WiSeqloss_WoChat_WoNoise \
       /work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_wochat
