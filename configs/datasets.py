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
    result_folder: str = None  
    
    

@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"
    data_path: str = "/work/valex1377/semi_at_llama/kd_datasets/alpaca_data_en_52k.json"   
    result_folder: str = None  
    
@dataclass
class iwslt2017deennat_dataset:
    dataset: str =  "iwslt2017deen_dataset"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"   
    src: str = 'de'
    tgt: str = 'en'   
    result_folder: str = None       
    
@dataclass
class openwebtext_kd_20k_dataset:
    dataset: str = "openwebtext_kd_20k_dataset"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"
    data_path: str = "/work/valex1377/semi_at_llama/kd_datasets/openwebtext/train.json"   
    result_folder: str = None   
    
@dataclass
class openwebtext_kd_80k_dataset:
    dataset: str = "openwebtext_kd_80k_dataset"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"
    data_path: str = "/work/valex1377/semi_at_llama/kd_datasets/openwebtext/train-80k.json"   
    result_folder: str = None        
    data_num: int = 20000
    
    
@dataclass
class mt_bench_dataset:
    dataset: str = "mt_bench_dataset"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"
    data_path: str = "/work/valex1377/semi_at_llama/kd_datasets/mt_bench/question_single.json"   
    result_folder: str = None        
