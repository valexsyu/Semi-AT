# python finetuning.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/llama-nat/llama_model/7B-chat \
#          --num_epochs 1 \
#          --batch_size_training 1 \


# torchrun --nnodes 1 --nproc_per_node 2  finetuning.py \
#          --dataset samsum_dataset \
#          --model_name /work/valex1377/llama-nat/llama_model/7B-chat \
#          --num_epochs 1 \
#          --enable_fsdp --pure_bf16 \
#          --dist_checkpoint_folder /work/valex1377/llama-nat/llama_model/7B-chat-ttt \
#          --batch_size_training 4 \        


# python generate_kd.py \
#          --model_name /work/valex1377/llama-nat/llama_model/7B-chat \
#          --seed 40 \
#          --output_dir "/work/valex1377/llama-nat/model_outputs/kd_data_tokenize_1.txt"

python generate_kd.py \
         --model_name /work/valex1377/llama-nat/llama_model/7B-chat \
         --seed 40 \
         --output_dir "/work/valex1377/llama-nat/model_outputs/gggg.txt"