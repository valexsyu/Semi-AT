import openai
from openai import OpenAI


import json
import argparse
import tqdm
import time

import numpy as np


argparser = argparse.ArgumentParser()
argparser.add_argument('--key', type=str, required=True)
argparser.add_argument('--model', type=str, default='gpt-4-0613')
args = argparser.parse_args()
openai.api_key = args.key

client = OpenAI(
    organization='org-9VP7zbu5OprKdttIEI0m2wqX',
)

questions = [
    'Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.',
    'Can you help me write a formal email to a potential business partner proposing a joint venture?',
    'Can you help me write a resignation letter to my current employer, while leaving on good terms and expressing gratitude for the opportunities provided?',
    'Use an appropriate format to structure a formal letter of recommendation for a student applying to a prestigious graduate program in computer science.',
    'Write a compelling product launch announcement email to inform our customers of our new software solution.',
    'Draft an apology email to a customer who experienced a delay in their order, and provide reassurance that the issue has been resolved.',
    'Write a script for a YouTube video exploring the history and cultural significance of jazz.',
    'Write a captivating movie review for a recently released science fiction film, discussing its plot, characters, and special effects.',
    'Structure a podcast script for an episode discussing the influence of streaming platforms on the music industry.',
    'Write a symphony concert review, discussing the orchestra\'s performance and overall audience experience.',
]

# model_A = [line[:-1] for line in open('7b-base-answer')]
model_A = [line[:-1] for line in open('st_openwebtext-full-80k-ar-1_epoch')]
model_B = [line[:-1] for line in open('st_openwebtext-full-20k-ar-3_epoch')]

verdicts = list()
for i in range(len(model_A)):
    cur_prompt = f"""
    [System]
    Please act as an impartial judge and evaluate the quality of the responses provided by two
    AI assistants to the user question displayed below. You should choose the assistant that
    follows the user’s instructions and answers the user’s question better. Your evaluation
    should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
    and level of detail of their responses. Begin your evaluation by comparing the two
    responses and provide a short explanation. Avoid any position biases and ensure that the
    order in which the responses were presented does not influence your decision. Do not allow
    the length of the responses to influence your evaluation. Do not favor certain names of
    the assistants. Be as objective as possible. After providing your explanation, output your
    final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
    if assistant B is better, and "[[C]]" for a tie.
    [User Question]
    {questions[i]}
    [The Start of Assistant A’s Answer]
    {model_A[i]}
    [The End of Assistant A’s Answer]
    [The Start of Assistant B’s Answer]
    {model_B[i]}
    [The End of Assistant B’s Answer]
    """


    _response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "system", "content": cur_prompt}],
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=1
    )
    time.sleep(0.5)

    verdict = _response.choices[0].message.content
    print(verdict)

    verdicts.append(verdict)
