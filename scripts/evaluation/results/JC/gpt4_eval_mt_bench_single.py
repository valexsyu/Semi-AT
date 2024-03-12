import openai
from openai import OpenAI
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
argparser.add_argument('--model_A', type=str)
args = argparser.parse_args()
openai.api_key = args.key

client = OpenAI(
    organization='org-9VP7zbu5OprKdttIEI0m2wqX',
)

encoding = tiktoken.encoding_for_model("gpt-4-0125-preview")


# Path to your .jsonl file
file_path = 'mt_bench/question.jsonl'

# List to hold each JSON object
data = []

# Open the .jsonl file and read each line
with open(file_path, 'r') as file:
    for line in file:
        # Parse the JSON object and append to the list
        data.append(json.loads(line))

questions = [doc['turns'][0] for doc in data if doc['category'] in ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']]
categories = [doc['category'] for doc in data if doc['category'] in ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']]

# breakpoint()

model_A = [line[:-1] for line in open(args.model_A)]

# Regex pattern to find the rating value
pattern = r"\[\[(\d{1,2})\]\]"

judgements = list()
verdicts = list()
prompt_token_counts = 0
response_token_counts = 0
for i in tqdm(range(len(model_A))):

    if categories[i] in ['reasoning', 'math']:
        cur_prompt = f"""[System]
        You are a helpful assistant.
        [Instruction]
        Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".
        
        [Question]
        {questions[i]}

        [The Start of Assistant A’s Answer]
        {model_A[i]}
        [The End of Assistant A’s Answer]"""
    else:
        cur_prompt = f"""[System]
        You are a helpful assistant.
        [Instruction]
        Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".
        
        [Question]
        {questions[i]}

        [The Start of Assistant A’s Answer]
        {model_A[i]}
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

pathA = args.model_A.split('/')[1]
np.save(f'mt_bench_result/eval_results_single/{pathA}.judgements', judgements)
np.save(f'mt_bench_result/eval_results_single/{pathA}.verdicts', verdicts)


verdicts = np.array(verdicts)


print(f'Average rating: {verdicts.mean()}')


# print prompt and response token counts
print(f'prompt token count: {prompt_token_counts}\n response token count: {response_token_counts}\n')
print(f'total cost = {prompt_token_counts / 1000 * 0.01 + response_token_counts / 1000 * 0.03}')

