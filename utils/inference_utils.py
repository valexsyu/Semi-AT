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



def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    from fairseq import tokenizer

    # Tokens are strings here
    hypo_tokens = tokenizer.tokenize_line(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = tokenizer.tokenize_line(src_str) + ["<eos>"]
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return " ".join(hypo_tokens)

def post_process_prediction(
    hypo_tokens,
    src_str,
    alignment,
    align_dict,
    tgt_dict,
    remove_bpe=None,
    extra_symbols_to_ignore=None,
):
    hypo_str = tgt_dict.string(
        hypo_tokens, remove_bpe, extra_symbols_to_ignore=extra_symbols_to_ignore
    )
    if align_dict is not None:
        hypo_str = replace_unk(
            hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string()
        )
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def inference_semi_at(approx_model, tokenizer, test_dataloader, inference_config, result_path , kwargs):

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
        if batch['labels'] is not None:
            batch_labels = batch['labels']
            batch.pop("labels",None)
            
        
        epoch_start_time = time.perf_counter()
        for key in batch.keys():
            batch[key] = batch[key].to('cuda:0')        
        with torch.no_grad():
            outputs = approx_model.generate(**batch, generation_config=inference_config,streamer=streamer, 
                                            max_new_tokens=inference_config.max_new_tokens) 

            
        epoch_end_time = time.perf_counter()-epoch_start_time  
        breakpoint()
        if result_path is not None:
            for _,(tgt_sent,hyp_sent) in enumerate(zip(batch_labels, outputs.tolist())):
                print("T-{}".format(tokenizer.decode((tgt_sent),skip_special_tokens=True)), file=result_path)  
                print("H-{}".format(tokenizer.decode(hyp_sent,skip_special_tokens=True)), file=result_path)                  
    return results



