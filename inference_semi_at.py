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
# from llama_recipes.configs import inference_config as inference_config
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from utils.inference_utils import generate_kd
from configs import SemiATGenerationConfig as INFERENCE_CONFIG
from dataclasses import dataclass, asdict

from semi_at_model import LlamaSemiATForCausalLM


def main(**kwargs):
    # Update the configuration for the training and sharding process
    inference_config = INFERENCE_CONFIG(**kwargs)
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(inference_config.seed)
    torch.manual_seed(inference_config.seed)
    



        
    
    target_model = LlamaSemiATForCausalLM.from_pretrained(
        inference_config.model_name,
        load_in_8bit=True if inference_config.quantization else None,
        device_map="auto" if inference_config.quantization else None,
        use_cache=inference_config.use_cache,
    )

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(inference_config.model_name)
    
    target_model.resize_token_embeddings(target_model.config.vocab_size + 2)  
    breakpoint()
    
    
    # tokenizer.add_special_tokens(
    #         {
    #             "mask_token": "<MASK>",
    #             "pad_token": "<PAD>",
    #         }
    #     )
    # target_model.resize_token_embeddings(target_model.config.vocab_size + 2)    

    # Prepare the model for int8 training if quantization is enabled
    if inference_config.quantization:
        target_model = prepare_model_for_int8_training(target_model)

    if inference_config.use_peft:
        peft_config = generate_peft_config(inference_config, kwargs)
        target_model = get_peft_model(target_model, peft_config)

    target_model.to("cuda")
    
    inference_kwargs = asdict(inference_config)
    inference_kwargs = remove_unused_kwargs(inference_kwargs)    
    
    # Start the training process
    results = generate_kd(
        target_model,
        tokenizer,
        inference_config,
        inference_kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)