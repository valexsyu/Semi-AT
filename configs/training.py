from llama_recipes.configs import train_config as TRAIN_CONFIG
from dataclasses import dataclass


@dataclass
class train_config(TRAIN_CONFIG):
    insert_mask_num: int=1
    model_name: str="/work/valex1377/semi_at_llama/model_outputs/LlamaSemiATForCausalLM"
    train_data_num: int=10000
    noise_rate: float=0.15
    semi_at_attention: bool=False
    insert_token_num: int=0
    semi_at_insert_token_id: int=2
    sequence_label_ingnore: bool=False
    target_kldiv_loss_enable: bool=False
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    llama_model: str=None
    run_validation: bool=True
    num_workers_dataloader: int=4
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False    
    output_dir: str = "/work/valex1377/semi_at_llama/model_outputs/PEFT/LlamaSemiATForCausalLM"
    freeze_layers: bool = False    
    num_freeze_layers: int = 1
    quantization: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="/work/valex1377/semi_at_llama/llama_model/KD_ft_b1_WoKLDiv_WoSeqloss_1" # will be used if using FSDP
    initialize_llamasemiat: bool = False
    

