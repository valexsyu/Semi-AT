# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class kd_dataset:
    dataset: str =  "kd_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    dataset_dir: str = "/work/valex1377/llama-nat/kd_datasets/kd_data_tokenize/"
    insert_token_num: int=0
    semi_at_insert_token_id: int=2
    
