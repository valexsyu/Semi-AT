# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
from contextlib import nullcontext


import torch
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer


from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.train_utils import save_train_params


def write_to_file(results, result_path , file_split='hypo' , file_prefix="" , file_type=".txt"):
    _path = os.path.join(result_path, file_split + file_prefix + file_type)
    with open(_path , 'w') as file:
        for item in results[file_split]:
            file.write('\n'.join(str(v) for v in item))


def generate_kd(target_model, tokenizer, inference_config ,inference_kwargs):

    autocast = torch.cuda.amp.autocast if inference_config.use_fp16 else nullcontext
    results = {}
    target_model.eval()
    generate_data_num = inference_kwargs.pop('generate_data_num', 4000)
    pbar = tqdm(colour="blue", desc=f"Genereate number:", total=generate_data_num, dynamic_ncols=True)
    for step in range(generate_data_num):
        batch = creat_batch(inference_config.batch_size,tokenizer.bos_token_id)
        
        for key in batch.keys():
            batch[key] = batch[key].to('cuda:0')
        with torch.no_grad():
            breakpoint()
            target_output = target_model.generate(**batch, **inference_kwargs)
            tensor_values = target_output.flatten().tolist()
            line = " ".join(map(str, tensor_values))
            with open(inference_config.output_dir, "a") as file:
                file.write(line + "\n" )

        pbar.update(1)
    return results


def creat_batch(size,bos) :
    return {
        "input_ids":torch.full((size,1), bos, dtype=torch.long),
        "attention_mask":torch.full((size,1), 1, dtype=torch.long),
    }




