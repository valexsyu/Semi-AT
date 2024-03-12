
import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import (
    default_data_collator,
)
import os
from tqdm import tqdm
import subprocess


from semi_at_model import LlamaSemiATForCausalLM
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)

from configs import SemiATGenerationConfig as INFERENCE_CONFIG
import gc


# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["llama2-7b"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["llama2-70b"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    parser.add_argument('--dataset_path', '-dp', type=str, default=None, help='dataset path')
    parser.add_argument('--result_folder', '-rf', type=str, default=None, help='test')
    parser.add_argument('--load_bits', '-lb', type=int, default=16, help='4 / 8 / 16 for approx model bits ')
    args = parser.parse_args()
    return args




def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def get_gpu_usage():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_usage, memory_usage = map(int, result.strip().split(', '))
    # print("gpu_usage:{} , memory_usage:{}".format(gpu_usage,memory_usage))
    return gpu_usage, memory_usage    
    
def benchmark(fn, print_prefix, batch, generation_config,streamer=None, 
                                            max_new_tokens=400):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    print("============START MEASURE TEST==========")
    gpu_usage=[] ; memory_usage=[]
    with contexttimer.Timer() as t:
        for _ in range(TEST_TIME): 
            output = fn(**batch, generation_config=generation_config, streamer=None, max_new_tokens=max_new_tokens) 
            gpu,memory=get_gpu_usage()
            gpu_usage.append(gpu)
            memory_usage.append(memory)                
    
    print("avg_gpu = {}".format(sum(gpu_usage)/TEST_TIME))
    print("avg_memory = {}".format(sum(memory_usage)/TEST_TIME))

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")











def test_semi_at_time(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,load_bits=16):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    delta_para_path = "/work/valex1377/semi_at_llama/llama_model/models_hf/LlamaSemiATForCausalLM_diff/delta.pt"
    max_new_tokens = num_tokens
    insert_token_num = 1
    inference_config = INFERENCE_CONFIG(model_name=approx_model_name,insert_token_num=insert_token_num,insert_token_id=32000,max_new_tokens=max_new_tokens,delta_para_path=delta_para_path)
    tokenizer = LlamaTokenizer.from_pretrained(approx_model_name)
    # Set the padding and eos.
    inference_config.pad_token_id = tokenizer.pad_token_id
    inference_config.bos_token_id = tokenizer.bos_token_id
    inference_config.eos_token_id = tokenizer.eos_token_id  
    print(f"begin loading models: \n {approx_model_name}")
    
    small_model = LlamaSemiATForCausalLM.from_pretrained(
        approx_model_name,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=None,
    )  


    diff_state_dict = torch.load(inference_config.delta_para_path)
        
    # Add the difference to the parameters of approx_model
    approx_model_state_dict = small_model.state_dict()
    for param_tensor in diff_state_dict:
        if param_tensor in approx_model_state_dict:
            approx_model_state_dict[param_tensor] += diff_state_dict[param_tensor]

    # Update the parameters of approx_model
    small_model.load_state_dict(approx_model_state_dict)    
    del diff_state_dict
    gc.collect()       
    
    small_model.to(torch_device)







               
    print("finish loading models")
    print("load in {} bits".format(load_bits))
    
    prompt =  tokenizer.encode(tokenizer.bos_token + input_text, add_special_tokens=False)
    
    example = torch.tensor(prompt, dtype=torch.int64)
    example_mask = example.ge(0)
    example[~example_mask] = 0   
    batch =  {
            "input_ids": example.unsqueeze(0),
            "attention_mask":example_mask.unsqueeze(0),
    }    
    for key in batch.keys():
        batch[key] = batch[key].to('cuda:0')            

    top_k = 20
    top_p = 0.9

    output = small_model.generate(**batch, generation_config=inference_config,streamer=None, 
                                            max_new_tokens=inference_config.max_new_tokens) 
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(small_model.generate, "AS_small", 
                  batch=batch, generation_config=inference_config,streamer=None, max_new_tokens=inference_config.max_new_tokens)



if __name__ == "__main__":
    args = parse_arguments()
    
    
    # generate_batch(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
    #          random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, dataset_path = args.dataset_path,
    #          result_folder=args.result_folder,load_bits=args.load_bits)    
    
    # generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
    #          random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)
    
    # test_at_time(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
    #          random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, load_bits=args.load_bits) 
    
    test_semi_at_time(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, load_bits=args.load_bits) 