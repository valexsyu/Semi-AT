# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
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
    default_data_collator,
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
    get_dataloader_kwargs,
)
from utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from utils.inference_utils import inference_semi_at
from utils.config_utils import generate_dataset_config
from configs import SemiATGenerationConfig as INFERENCE_CONFIG
from dataclasses import dataclass, asdict

from semi_at_model import LlamaSemiATForCausalLM


def main(**kwargs):
    # Update the configuration for the training and sharding process
    inference_config = INFERENCE_CONFIG(**kwargs)
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(inference_config.seed)
    torch.manual_seed(inference_config.seed)
    
    approx_model = LlamaSemiATForCausalLM.from_pretrained(
        inference_config.model_name,
        load_in_8bit=True if inference_config.quantization else None,
        device_map="auto" if inference_config.quantization else None,
        use_cache=inference_config.use_cache,
    )

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(inference_config.model_name)
    
    # Set the padding and eos.
    inference_config.pad_token_id = tokenizer.pad_token_id
    inference_config.bos_token_id = tokenizer.bos_token_id
    inference_config.eos_token_id = tokenizer.eos_token_id
    
    

    # Prepare the model for int8 training if quantization is enabled
    if inference_config.quantization:
        approx_model = prepare_model_for_int8_training(approx_model)

    if inference_config.use_peft:
        peft_config = generate_peft_config(inference_config, kwargs)
        approx_model = get_peft_model(approx_model, peft_config)

    approx_model.to("cuda")
    
    dataset_config = generate_dataset_config(inference_config, kwargs)

     # Load and preprocess the dataset for testing
    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split=inference_config.dataset_split,
    )
    test_sampler = None
    # Create DataLoaders for the testing and validation dataset
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=inference_config.batch_size_testing,
        num_workers=inference_config.num_workers_dataloader,
        pin_memory=True,
        sampler=test_sampler if test_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )    

    if dataset_config.result_folder is not None:
        result_path = os.path.join(inference_config.model_name, dataset_config.result_folder)
        os.makedirs(result_path, exist_ok=True)
        output_path = os.path.join(
            result_path,
            "generate-{}.txt".format(inference_config.dataset_split),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            # Start the training process
            results = inference_semi_at(
                approx_model,
                tokenizer,
                test_dataloader,
                inference_config,
                h,
                kwargs,
            )
    else:
        results = inference_semi_at(
            approx_model,
            tokenizer,
            test_dataloader,
            inference_config,
            None,#sys.stdout,
            kwargs,
        )        
        
        
  
    



if __name__ == "__main__":
    fire.Fire(main)