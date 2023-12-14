# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar
import sys


from transformers.generation.configuration_utils import GenerationConfig


@dataclass
class inference_config():
    model_name: str="PATH/to/LLAMA/7B"
    use_fp16: bool=False
    dataset = "samsum_dataset"
    use_peft: bool=False
    peft_method: str = "lora" # None , llama_adapter, prefix
    peft_model: str = None
    output_scores: bool = True
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    upsampling_rate : int=1
    nat_atten_mask: bool = False
    special_decode: bool = False
    more_step: int = 0
    result_file_prefix : str = ""
    return_dict_in_generate: bool=True
    max_length: int=150
    max_new_tokens: int=100
    # num_beams: int = 5
    do_sample: bool = True   
    use_cache: bool = True
    

@dataclass
class inference_kd_config():
    model_name: str="PATH/to/LLAMA/7B"
    seed: int=42
    quantization: bool=False
    use_fp16: bool=False
    dataset = "samsum_dataset"
    use_peft: bool=False
    peft_method: str = "lora" # None , llama_adapter, prefix
    peft_model: str = None
    output_scores: bool = False
    use_fast_kernels: bool = True # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    result_file_prefix : str = ""
    return_dict_in_generate: bool=False
    max_new_tokens: int=200
    num_beams: int = 5
    do_sample: bool = True   
    use_cache: bool = True    
    batch_size: int = 1
    output_dir : str= "/work/valex1377/llama-nat/model_outputs/kd_data_tokenize_valid.txt"
    generate_data_num : int=200
    
    



    
