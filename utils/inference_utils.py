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
from transformers import LlamaTokenizer, TextStreamer


from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.train_utils import save_train_params


def write_to_file(results, result_path , file_split='hypo' , file_prefix="" , file_type=".txt"):
    _path = os.path.join(result_path, file_split + file_prefix + file_type)
    with open(_path , 'w') as file:
        for item in results[file_split]:
            file.write('\n'.join(str(v) for v in item))


def generate_kd(approx_model, tokenizer, inference_config ,inference_kwargs):

    autocast = torch.cuda.amp.autocast if inference_config.use_fp16 else nullcontext
    results = {}
    approx_model.eval()
    generate_data_num = inference_kwargs.pop('generate_data_num', 4000)
    pbar = tqdm(colour="blue", desc=f"Genereate number:", total=generate_data_num, dynamic_ncols=True)
    for step in range(generate_data_num):
        batch = creat_batch(inference_config.batch_size,tokenizer.bos_token_id)
        
        for key in batch.keys():
            batch[key] = batch[key].to('cuda:0')
        with torch.no_grad():
            
            target_output = approx_model.generate(**batch, **inference_kwargs)
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


def inference_semi_at(approx_model, tokenizer, test_dataloader, inference_config, kwargs):

    autocast = torch.cuda.amp.autocast if inference_config.use_fp16 else nullcontext
    if inference_config.insert_token_num == 0 :
        streamer = TextStreamer(tokenizer=tokenizer)
    else:
        streamer = None
    epoch_times = []
    results = {}
    output_token = []
    approx_model.eval()
    pbar = tqdm(test_dataloader,colour="blue", desc=f"Testing num: ", total=len(test_dataloader), dynamic_ncols=True)
    for step, batch in enumerate(test_dataloader):    
        
        epoch_start_time = time.perf_counter()
        for key in batch.keys():
            batch[key] = batch[key].to('cuda:0')        
        with torch.no_grad():
            outputs = approx_model.generate(**batch, generation_config=inference_config,streamer=streamer, 
                                            max_new_tokens=inference_config.max_new_tokens) 
            
        epoch_end_time = time.perf_counter()-epoch_start_time               
        breakpoint()
        print("==================================END============================================")
    return results



