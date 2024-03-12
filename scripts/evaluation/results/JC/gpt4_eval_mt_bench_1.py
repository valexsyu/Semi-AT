import openai
from openai import OpenAI


import json
import argparse
import tqdm
import time


argparser = argparse.ArgumentParser()
argparser.add_argument('--key', type=str, required=True)
argparser.add_argument('--model', type=str, default='gpt-4-0613')
args = argparser.parse_args()
openai.api_key = args.key

client = OpenAI(
    organization='org-9VP7zbu5OprKdttIEI0m2wqX',
)


cur_prompt = """
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
Can you help me write a formal email to a potential business partner proposing a joint venture?
[The Start of Assistant A’s Answer]
Of course, I'd be happy to help you write a formal email to a potential business partner proposing a joint venture! Here's an example of a formal email that you could use as a starting point:\nSubject: Proposal for Joint Venture\nDear [Potential Business Partner's Name],\nI hope this email finds you well. I am writing to propose a joint venture between our two companies, [Your Company Name] and [Potential Business Partner's Name]. Our goal is to create a mutually beneficial relationship that leverages our unique strengths and expertise to create a new opportunities.\nWe have carefully considered your proposal and found that we have several areas of common interest and collaboration. Our teams believe that a joint venture between our two companies would be highly beneficial, as it would allow us to leverage our resources, expertise, and knowledge to create a joint product or service that meets the needs of both parties.\nI would like to discuss this proposal further with you and explore the potential of a successful joint venture. Please let me know your availability for a call or meeting to discuss this further.\nThank you for your time and consideration.\nBest regards,\n[Your Name]\n\nPlease feel free to modify this email according to your specific needs and the details of your potential business partner. Remember to always be respectful, honest, and clear in your communication, and to ensure that any partnership is mutually beneficial and positive in nature.
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
Of course, I can help you write a formal email to a potential business partner proposing a joint venture. Here's an example:\n\nDear [Name of the potential business partner],\n\nI hope this email finds you well. I am writing to you with a proposal for a joint venture. I believe that our companies share similar values and goals, and that a collaboration between us could be mutually beneficial.\n\nI would like to propose that we work together on a project that combines our strengths. I am confident that we could achieve great success if we work together.\n\nI would be happy to discuss the details of the project with you. Please let me know if you are interested.
[The End of Assistant B’s Answer]
"""


_response = client.chat.completions.create(
    model=args.model,
    messages=[{"role": "system", "content": cur_prompt}],
    temperature=1,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    n=1
)
time.sleep(0.5)

verdict = _response.choices[0].message.content
print(verdict)
