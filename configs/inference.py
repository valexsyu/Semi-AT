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
    output_dir : str= "/work/valex1377/semi_at_llama/model_outputs"
    generate_data_num : int=200
    
 


class SemiATGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):            
        super().__init__(**kwargs)
        self.model_name = kwargs.pop("model_name", None) 
        self.seed = kwargs.pop("seed", 42)
        self.insert_token_num = kwargs.pop("insert_token_num", 1)
        self.quantization = kwargs.pop("quantization", False)
        self.dataset = kwargs.pop("dataset", "samsum_dataset")
        self.use_peft = kwargs.pop("use_peft", False)
        self.peft_method = kwargs.pop("peft_method", "lora")
        self.use_fast_kernels = kwargs.pop("use_fast_kernels", True) 
        self.batch_size_testing = kwargs.pop("batch_size_testing", 1) 
        self.num_workers_dataloader = kwargs.pop("num_workers_dataloader", 1) 
        self.use_fp16 = kwargs.pop("use_fp16", False) 
        self.dataset_split = kwargs.pop("dataset_split", "test") 
        self.insert_token_id = kwargs.pop("insert_token_id", 0) 
        self.max_new_tokens = kwargs.pop("max_new_tokens", 400) 
        self.quantization = kwargs.pop("quantization", False) 
        self.insert_waiting = kwargs.pop("insert_waiting", False) 
        self.add_diff_para = kwargs.pop("add_diff_para", False) 
        self.no_chat_model_path = kwargs.pop("no_chat_model_path", None) 
        self.chat_model_path = kwargs.pop("chat_model_path", None) 
        self.delta_para_path = kwargs.pop("delta_para_path", None) 
        self.debug = kwargs.pop("debug", False) 
        
        
        
    
    

    




    
