import openai
from openai import OpenAI
import os
import tiktoken

import json
import argparse
from tqdm import tqdm
import time

import numpy as np
import re


argparser = argparse.ArgumentParser()
argparser.add_argument('--key', type=str, required=True)
argparser.add_argument('--eval_model', type=str, default='gpt-4-0125-preview')
argparser.add_argument('--question_path', type=str, default='/work/valex1377/semi_at_llama/kd_datasets/mt_bench/question_single.json')
argparser.add_argument('--datapath_A', type=str, default='/work/valex1377/semi_at_llama/llama_model/models_hf/KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_nochat/mt_bench_dataset/insert_waiting_add_diff_para/generate-test.txt')
argparser.add_argument('--datapath_B', type=str, default='/work/valex1377/semi_at_llama/llama_model/models_hf/llama-2-7b-chat/mt_bench_dataset/at/generate-test.txt')
args = argparser.parse_args()
openai.api_key = args.key

client = OpenAI(
    organization='org-9VP7zbu5OprKdttIEI0m2wqX',
)

encoding = tiktoken.encoding_for_model(args.eval_model)

# questions = [
#     'Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.',
#     'Can you help me write a formal email to a potential business partner proposing a joint venture?',
#     'Can you help me write a resignation letter to my current employer, while leaving on good terms and expressing gratitude for the opportunities provided?',
#     'Use an appropriate format to structure a formal letter of recommendation for a student applying to a prestigious graduate program in computer science.',
#     'Write a compelling product launch announcement email to inform our customers of our new software solution.',
#     'Draft an apology email to a customer who experienced a delay in their order, and provide reassurance that the issue has been resolved.',
#     'Write a script for a YouTube video exploring the history and cultural significance of jazz.',
#     'Write a captivating movie review for a recently released science fiction film, discussing its plot, characters, and special effects.',
#     'Structure a podcast script for an episode discussing the influence of streaming platforms on the music industry.',
#     'Write a symphony concert review, discussing the orchestra\'s performance and overall audience experience.',
# ]


# Initialize variables

hypothesis = ""
all_hypotheses = [[], []]
question=[]
n_values = [[], []]
i_values = [[], []]
ratios = [[], []]
start=False
num=0
if args.datapath_B == "None":
    num_model = 1
    args.datapath_B = args.datapath_A
else:
    num_model = 2
datapaths = [args.datapath_A, args.datapath_B] 

# Read the file
with open(args.datapath_A, 'r') as file_A:
    with open(args.datapath_B, 'r') as file_B:
        files = [file_A, file_B]
        for i in range(num_model):
            for index, line in enumerate(files[i]):
                if line.startswith('H-'):
                    num = num+1
                    start = True
                    hypothesis += line[2:] + " "
                    # hypothesis += line[2:] + " "
                elif line.startswith('E-') or line.startswith('I-') or line.startswith('N-'):
                    # Add the target and hypothesis_A to the lists
                    if line.startswith('N-'):
                        n_values[i].append(float(line[2:].strip()))
                    elif line.startswith('I-'):
                        i_values[i].append(float(line[2:].strip()))
                        if n_values[i]:
                            # Calculate the ratio of the last "N-" value to this "I-" value
                            ratios[i].append(n_values[i][-1] / i_values[i][-1])
                    elif line.startswith('E-'):
                        all_hypotheses[i].append(hypothesis)
                        hypothesis = ""
                        start = False
                elif start :
                    hypothesis += line[:]                    
                    
                if line.startswith('N-'):
                    n_values[i].append(float(line[2:].strip()))
                elif line.startswith('I-'):
                    i_values[i].append(float(line[2:].strip()))
                    if n_values[i]:
                        # Calculate the ratio of the last "N-" value to this "I-" value
                        ratios[i].append(n_values[i][-1] / i_values[i][-1])
                
# Calculate the average of all "N-" and "I-" values
for i in range(len(n_values)):
    if len(n_values[i]) < 1 :
        break
    print("============speed==============")
    average_n = sum(n_values[i]) / len(n_values[i]) if n_values[i] else 0
    average_i = sum(i_values[i]) / len(i_values[i]) if i_values[i] else 0
    average_speed = (average_n/average_i)*100 if average_i else 0
    print("avg n :{}".format(average_n))
    print("avg i :{}".format(average_i))
    print("avg speed :{}".format(average_speed))

ann = json.load(open(args.question_path))
all_questions = []
all_category = []
for i, line in enumerate(ann):
    all_questions.append(line['turns'])
    all_category.append(line['category'])


# # model_A = [line[:-1] for line in open('7b-base-answer')]
# model_A = [line[:-1] for line in open('st_openwebtext-full-80k-ar-1_epoch')]
# model_B = [line[:-1] for line in open('st_openwebtext-full-20k-ar-3_epoch')]

# Regex pattern to find the rating value
pattern = r"\[\[(\d{1,2})\]\]"
judgements = list()
verdicts = list()
prompt_token_counts = 0
response_token_counts = 0



for j in range(num_model) :
    judgements = list()
    verdicts = list()
    prompt_token_counts = 0
    response_token_counts = 0    
    for i in tqdm(range(len(all_questions))):
    # for i in range(5,7):
        if all_category[i] in ['reasoning', 'math']:
            cur_prompt = f"""[System]
            You are a helpful assistant.
            [Instruction]
            Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".
            
            [Question]
            {all_questions[i]}

            [The Start of Assistant A’s Answer]
            {all_hypotheses[j][i]}
            [The End of Assistant A’s Answer]"""
        else:
            cur_prompt = f"""[System]
            You are a helpful assistant.
            [Instruction]
            Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".
            
            [Question]
            {all_questions[i]}

            [The Start of Assistant A’s Answer]
            {all_hypotheses[j][i]}
            [The End of Assistant A’s Answer]"""


        prompt_token_counts += len(encoding.encode(cur_prompt))

        _response = client.chat.completions.create(
            model=args.eval_model,
            messages=[{"role": "system", "content": cur_prompt}],
            temperature=0.0,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            n=1
        )
        
        time.sleep(0.5)

        judgement = _response.choices[0].message.content

        response_token_counts += len(encoding.encode(judgement))

        judgements.append(judgement)

        # Search for the pattern in the text
        match = re.search(pattern, judgement)

        # Extract the rating value if a match is found
        rating = int(match.group(1)) if match else 0

        verdicts.append(rating)


    parent_dir = os.path.dirname(datapaths[j])
    # create the file path for verdicts.txt in the same directory
    judgements_file_path = os.path.join(parent_dir, "single_judgements.txt")
    verdicts_file_path = os.path.join(parent_dir, "single_verdicts.txt")
    np.savetxt(judgements_file_path, judgements, fmt='%s')
    np.savetxt(verdicts_file_path, verdicts, fmt='%s')


    verdicts = np.array(verdicts)
    print("======================================================================================")
    print(datapaths[j])
    print(f'Average rating: {verdicts.mean()}')


    # print prompt and response token counts
    print(f'prompt token count: {prompt_token_counts}\n response token count: {response_token_counts}\n')
    print(f'total cost = {prompt_token_counts / 1000 * 0.01 + response_token_counts / 1000 * 0.03}')
    


