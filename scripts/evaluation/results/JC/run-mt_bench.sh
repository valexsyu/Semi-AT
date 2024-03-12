
export OPENAI_API_KEY=YOUR_KEY

# 第一次使用時執行
# curl https://api.openai.com/v1/models \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -H "OpenAI-Organization: org-9VP7zbu5OprKdttIEI0m2wqX"


python -i gpt4_eval_mt_bench.py \
    --key $OPENAI_API_KEY \

