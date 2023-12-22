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
    get_dataloader_kwargs,
)

from llama_recipes.utils.train_utils import (
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from utils.train_utils import train_kd
from utils.config_utils import generate_dataset_config

from configs import train_config as TRAIN_CONFIG
from configs import inference_config as INFERENCE_CONFIG
from utils.dataset_utils import get_preprocessed_dataset
from dataclasses import dataclass, asdict
from semi_at_model import LlamaSemiATForCausalLM


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
    kwargs.pop("noise_rate",None) 
    # kwargs.pop("semi_at_insert_token_id",None)
    
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

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name, padding="max_length", truncation=True)
    tokenizer.add_special_tokens(
            {
                "mask_token": "<mask>",
                "pad_token": "<PAD>",
            }
        )
    
    semi_at_insert_token_id = tokenizer.convert_tokens_to_ids('<mask>')
    update_config(train_config,**{"semi_at_insert_token_id":semi_at_insert_token_id})
    
    
    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        
        llama_config = LlamaConfig.from_pretrained(train_config.model_name)
        llama_config.use_cache = use_cache           
        if rank == 0:
            # target_model = LlamaForCausalLM.from_pretrained(
            #     train_config.model_name,
            #     load_in_8bit=True if train_config.quantization else None,
            #     device_map="auto" if train_config.quantization else None,
            #     use_cache=use_cache,
            # )   
              
            approx_model = LlamaSemiATForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )     
                            
              
            
            # approx_model = LlamaSemiATForCausalLM(llama_config, semi_at_insert_token_id)
            # approx_model.load_state_dict(target_model.state_dict())             
        else:
             
            with torch.device("meta"):
                # target_model = LlamaForCausalLM(llama_config)
                approx_model = LlamaSemiATForCausalLM(llama_config, semi_at_insert_token_id)
                # approx_model.load_state_dict(target_model.state_dict())   
                
        print("====================Load Model Finish=========================")    

    else:
        llama_config = LlamaConfig.from_pretrained(train_config.model_name)
        # target_model = LlamaForCausalLM.from_pretrained(
        #     train_config.model_name,
        #     load_in_8bit=True if train_config.quantization else None,
        #     device_map="auto" if train_config.quantization else None,
        #     use_cache=use_cache,
        # )
        
        approx_model = LlamaSemiATForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )        
        
        # approx_model = LlamaSemiATForCausalLM(llama_config, semi_at_insert_token_id)
        # approx_model.load_state_dict(target_model.state_dict())
    
    

    # # Save the model
    # approx_model.save_pretrained("/work/valex1377/llama-nat/model_outputs/LlamaSemiATForCausalLM")

    # breakpoint()
    # # Assuming your tokenizer is named 'tokenizer'
    # tokenizer.save_pretrained("/work/valex1377/llama-nat/model_outputs/LlamaSemiATForCausalLM")

    
    

    
    if not train_config.target_kldiv_loss_enable :
        target_model = None
        
    print("====================Load Model Finish=========================")
    
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            approx_model = BetterTransformer.transform(approx_model)
            if train_config.target_kldiv_loss_enable :
                target_model = BetterTransformer.transform(target_model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")


    approx_model.resize_token_embeddings(approx_model.config.vocab_size + 2)  
    if train_config.target_kldiv_loss_enable :  
        target_model.resize_token_embeddings(target_model.config.vocab_size + 2) 
    
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    print_model_size(approx_model, train_config, rank if train_config.enable_fsdp else 0)
    if train_config.target_kldiv_loss_enable :
        print_model_size(target_model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        approx_model = prepare_model_for_int8_training(approx_model)
        if train_config.target_kldiv_loss_enable :
            target_model = prepare_model_for_int8_training(target_model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        approx_model.to(torch.bfloat16)
        if train_config.target_kldiv_loss_enable :
            target_model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        approx_model = get_peft_model(approx_model, peft_config)
        approx_model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(approx_model, LlamaDecoderLayer)

        approx_model = FSDP(
            approx_model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if train_config.target_kldiv_loss_enable :
            target_model = FSDP(
                target_model,
                auto_wrap_policy= wrapping_policy,
                cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
                mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
                sharding_strategy=fsdp_config.sharding_strategy,
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
                sync_module_states=train_config.low_cpu_fsdp,
                param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
                if train_config.low_cpu_fsdp and rank != 0 else None,
            )        
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(approx_model)
            if train_config.target_kldiv_loss_enable :
                apply_fsdp_checkpointing(target_model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        approx_model.to("cuda")
        if train_config.target_kldiv_loss_enable :
            target_model.to("cuda")
    
    print("===============FSDP Model Finish=================")
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_config.semi_at_insert_token_id = semi_at_insert_token_id

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

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            approx_model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            approx_model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    
    inference_kwargs = asdict(inference_config)
    inference_kwargs = remove_unused_kwargs(inference_kwargs)    
    
     
    # Start the training process
    results = train_kd(
        approx_model,
        target_model,
        train_dataloader,
        eval_dataloader,        
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        inference_kwargs,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
