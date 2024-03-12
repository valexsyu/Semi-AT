max_tokens_list=(800)


load_bits=8
echo "=========================$load_bits bits==============================="
for max_token in "${max_tokens_list[@]}"
do
    python /work/valex1377/semi_at_llama/eval_semiar_time.py \
        --input "More than a million displaced Palestinians - about half of the Strip's population - are " \
        --target_model_name /work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat \
        --approx_model_name /work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_wochat \
        -b \
        --max_tokens $max_token \
        --load_bits $load_bits 
done

