## fsdp + ft
python finetuning_kd.py \
         --dataset kd_dataset \
         --pure_bf16 \
         --model_name /work/valex1377/llama-nat/llama_model/7B-chat \
         --num_epochs 1 \
         --batch_size_training 2 \
         --gradient_accumulation_steps 3 \
         --batching_strategy padding \
         --insert_token_num 1 \
         --sequence_label_ingnore \
          


