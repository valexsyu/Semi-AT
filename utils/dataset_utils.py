# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
from functools import partial
from pathlib import Path

import torch

from kd_datasets import (
    get_kd_dataset,
    get_samsum_dataset,
    get_alpaca_dataset,
    get_iwslt17deen_dataset,
    get_openwebtext_kd_20k_dataset,
    get_openwebtext_kd_80k_dataset,
    get_mt_bench_dataset,
    
)





def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e


DATASET_PREPROC = {
    "kd_dataset": get_kd_dataset,
    "samsum_dataset": get_samsum_dataset,
    "alpaca_dataset" : get_alpaca_dataset,
    "iwslt2017deennat_dataset" : get_iwslt17deen_dataset,
    "openwebtext_kd_20k_dataset" : get_openwebtext_kd_20k_dataset,
    "openwebtext_kd_80k_dataset" : get_openwebtext_kd_80k_dataset,
    "mt_bench_dataset" : get_mt_bench_dataset,
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        if split ==  "train":
            return dataset_config.train_split 
        elif split == "validation":
            return dataset_config.valid_split
        elif split == "test": 
            return dataset_config.test_split
        else:
            raise ValueError(f"please check the dataset config includle the file{split}")
        # return (
        #     dataset_config.train_split
        #     if split == "train"
        #     else dataset_config.test_split
        # )

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
