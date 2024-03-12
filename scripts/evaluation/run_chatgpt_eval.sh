
export OPENAI_API_KEY=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

# # 第一次使用時執行
# curl https://api.openai.com/v1/models \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -H "OpenAI-Organization: org-9VP7zbu5OprKdttIEI0m2wqX"

ROOT_PATH=/work/valex1377/semi_at_llama/llama_model/models_hf
DataPathA=$ROOT_PATH/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_wochat/mt_bench_dataset/add_diff_para/generate-test.txt
# DataPathA=$ROOT_PATH/KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat/mt_bench_dataset/insert_waiting_add_diff_para/generate-test.txt
# DataPathA=$ROOT_PATH/KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat_WoNoise/mt_bench_dataset/insert_waiting_add_diff_para/generate-test.txt
# DataPathA=$ROOT_PATH/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_wochat/mt_bench_dataset/insert_waiting_add_diff_para/generate-test.txt

# DataPathA=$ROOT_PATH/llama-2-7b-chat/mt_bench_dataset/speculative/4/generate-test.txt
# DataPathA=$ROOT_PATH/llama-68m/JackFram_llama-68m/speculative/generate-test.txt


# DataPathB=$ROOT_PATH/llama-2-7b-chat/mt_bench_dataset/at/generate-test.txt
DataPathB=$ROOT_PATH/llama-68m/JackFram_llama-68m/speculative/generate-test.txt

EVAL_MODEL=gpt-4-0125-preview
python /work/valex1377/semi_at_llama/scripts/evaluation/gpt4_eval_mt_bench.py \
    --key $OPENAI_API_KEY \
    --datapath_A $DataPathA \
    --datapath_B $DataPathB \
    --eval_model $EVAL_MODEL \

# python /work/valex1377/semi_at_llama/scripts/evaluation/gpt4_eval_mt_bench_single_rating.py \
#     --key $OPENAI_API_KEY \
#     --datapath_A $DataPathA \
#     --datapath_B None \
#     --eval_model $EVAL_MODEL 



