# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
)

from llama_recipes.utils.train_utils import (
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)

from configs import train_config as TRAIN_CONFIG
from configs import inference_config as INFERENCE_CONFIG
from utils.dataset_utils import get_preprocessed_dataset
from utils.config_utils import generate_dataset_config,get_dataloader_kwargs


from dataclasses import dataclass, asdict


def remove_unused_kwargs(kwargs):
    kwargs.pop("model_name",None)
    kwargs.pop("use_fp16",None)
    kwargs.pop("use_peft",None)
    kwargs.pop("peft_method",None)
    kwargs.pop("peft_model",None)
    kwargs.pop("quantization",None)
    kwargs.pop("use_fast_kernels",None)
    kwargs.pop("result_file_prefix",None)  
    kwargs.pop("upsampling_rate",None) 
    kwargs.pop("special_decode",None) 
    kwargs.pop("more_step",None) 
    kwargs.pop("nat_atten_mask",None) 
    return kwargs

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config , inference_config= TRAIN_CONFIG(), FSDP_CONFIG(), INFERENCE_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    update_config(inference_config, **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None


    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
            {
                "mask_token": "<MASK>",
                "pad_token": "<PAD>",
            }
        )
    
    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")
    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
    for step, batch in enumerate(train_dataloader):
        
        breakpoint()
        print("eee")

if __name__ == "__main__":
    fire.Fire(main)
