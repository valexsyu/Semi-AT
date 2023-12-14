from llama_recipes.configs import train_config as TRAIN_CONFIG
from dataclasses import dataclass


@dataclass
class train_config(TRAIN_CONFIG):
    insert_mask_num: int=1
    model_name: str="llama-nat/llama_model/7B-chat"
    train_data_num: int=10000
    noise_rate: float=0.15
    semi_at_attention: bool=False
    insert_token_num: int=0
    semi_at_insert_token_id: int=2
    sequence_label_ingnore: bool=False
    target_kldiv_loss_enable: bool=False
    
    
    

