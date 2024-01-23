# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class kd_dataset:
    dataset: str =  "kd_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    dataset_dir: str = "/work/valex1377/semi_at_llama/kd_datasets/kd_data_tokenize"
    insert_token_num: int=0
    semi_at_insert_token_id: int=2
    

@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"
    
    

@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"
    data_path: str = "/work/valex1377/semi_at_llama/kd_datasets/alpaca_data_en_52k.json"    