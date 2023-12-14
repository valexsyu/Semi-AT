import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
import os

# model_name="/work/valex1377/llama/models_hf/7B"
# model_name="/work/valex1377/llama/models_hf/7B-chat"
model_name="/work/valex1377/llama/models_hf/chat_nat_ft_batch_padloss_e7"
upsampling_rate=1
use_cache=True
nat_atten_mask=False
special_decode=True
more_step=2
output_scores=False
return_dict_in_generate=False
tokenizer = LlamaTokenizer.from_pretrained(model_name)
# model =LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
model =LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=False, device_map='auto')

# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/model_lora_e7_up1")
# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/nat_lora_e1")
# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/at_lora_e1")
# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/nat0_lora_e1")
# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/at_lora_e7-last")
# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/nat0_lora_e7-last")
# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/nat_lora_e7-last")
# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/nat_lora_e7_u2-last")
# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/chat_at_lora_e1-last")
# model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/chat_at_lora_e7-last")
# # model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/chat_nat_lora_e1-last")
# model = PeftModel.from_pretrained(model, "/work/valex1377/llama/PEFT/chat_nat_lora_e7-last")

# # eval_prompt = """
# # Summarize this dialog:
# # A: Hi Tom, are you busy tomorrow’s afternoon?
# # B: I’m pretty sure I am. What’s up?
# # A: Can you go with me to the animal shelter?.
# # B: What do you want to do?
# # A: I want to get a puppy for my son.
# # B: That will make him so happy.
# # A: Yeah, we’ve discussed it many times. I think he’s ready now.
# # B: That’s good. Raising a dog is a tough issue. Like having a baby ;-)
# # A: I'll get him one of those little dogs.
# # B: One that won't grow up too big;-)
# # A: And eat too much;-))
# # B: Do you know which one he would like?
# # A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
# # B: I bet you had to drag him away.
# # A: He wanted to take it home right away ;-).
# # B: I wonder what he'll name it.
# # A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
# # ---
# # Summary:
# # """
# eval_prompt = """
# Summarize this dialog:
# A: Hi Tom, are you busy tomorrow’s afternoon? B: I’m pretty sure I am. What’s up? A: Can you go with me to the animal shelter?. B: What do you want to do? A: I want to get a puppy for my son. B: That will make him so happy. A: Yeah, we’ve discussed it many times. I think he’s ready now. B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) A: I'll get him one of those little dogs. B: One that won't grow up too big;-) A: And eat too much;-)) B: Do you know which one he would like? A: Oh, yes, I took him there last Monday. He showed me one that he really liked. B: I bet you had to drag him away. A: He wanted to take it home right away ;-). B: I wonder what he'll name it. A: He said he’d name it after his dead hamster – Lemmy - he's a great Motorhead fan :-)))
# ---
# Summary:
# """

# model.eval()


# # model.generate(**model_input, max_new_tokens=200)
# # with torch.no_grad():
# #     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

def remove_duplicate_tokens(tokens):
    # remove_duplicate_tokens = torch.unique_consecutive(tokens)
    remove_duplicate_tokens = tokens
    return remove_duplicate_tokens
    

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")  
# with torch.no_grad():
#     model_input["nat_atten_mask"] = nat_atten_mask

    
#     print(tokenizer.decode(remove_duplicate_tokens(model.generate(**model_input ,max_new_tokens=100,upsampling_rate=upsampling_rate,\
#         use_cache=use_cache,special_decode=special_decode,more_step=more_step,output_scores=output_scores,return_dict_in_generate=return_dict_in_generate)[0]), skip_special_tokens=True))
#     print("Ans: \n A will go to the animal shelter tomorrow to get a puppy for her son. They already visited the shelter last Monday and the son chose the puppy.")


# eval_prompt = """
# Summarize this dialog:
# Emma: I’ve just fallen in love with this advent calendar! Awesome! I wanna one for my kids! Rob: I used to get one every year as a child! Loved them! Emma: Yeah, i remember! they were filled with chocolates! Lauren: they are different these days! much more sophisticated! Haha! Rob: yeah, they can be fabric/ wooden, shop bought/ homemade, filled with various stuff Emma: what do you fit inside? Lauren: small toys, Christmas decorations, creative stuff, hair bands & clips, stickers, pencils & rubbers, small puzzles, sweets Emma: WOW! That’s brill! X Lauren: i add one more very special thing as well- little notes asking my children to do something nice for someone else Rob: i like that! My sister adds notes asking her kids questions about christmas such as What did the 3 wise men bring? etc Lauren: i reckon it prepares them for Christmas Emma: and makes it more about traditions and being kind to other people Lauren: my children get very excited every time they get one! Emma: i can see why! :)
# ---
# Summary:
# """    
# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# with torch.no_grad():
#     model_input["nat_atten_mask"] = nat_atten_mask

#     print(tokenizer.decode(remove_duplicate_tokens(model.generate(**model_input, max_new_tokens=100,upsampling_rate=upsampling_rate,\
#         use_cache=use_cache,special_decode=special_decode,more_step=more_step,output_scores=output_scores,return_dict_in_generate=return_dict_in_generate)[0]), skip_special_tokens=True))
#     print("Ans: \n Emma and Rob love the advent calendar. Lauren fits inside calendar various items, for instance, small toys and Christmas decorations. Her children are excited whenever they get the calendar.")
    


# eval_prompt = """
# Summarize this dialog:
# Laura: I need a new printer :/ Laura: thinking about this one Laura: <file_other> Jamie: you're sure you need a new one? Jamie: I mean you can buy a second hand one Laura: could be
# ---
# Summary:
# """    
# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# with torch.no_grad():
#     model_input["nat_atten_mask"] = nat_atten_mask

#     print(tokenizer.decode(remove_duplicate_tokens(model.generate(**model_input, max_new_tokens=100,upsampling_rate=upsampling_rate,\
#         use_cache=use_cache,special_decode=special_decode,more_step=more_step,output_scores=output_scores,return_dict_in_generate=return_dict_in_generate)[0]), skip_special_tokens=True))
#     print("Ans: \n Laura is going to buy a printer.")
    
eval_prompt = """
Summarize this dialog:
Hannah: Hey, do you have Betty's number? Amanda: Lemme check Hannah: <file_gif> Amanda: Sorry, can't find it. Amanda: Ask Larry Amanda: He called her last time we were at the park together Hannah: I don't know him well Hannah: <file_gif> Amanda: Don't be shy, he's very nice Hannah: If you say so.. Hannah: I'd rather you texted him Amanda: Just text him 🙂 Hannah: Urgh.. Alright Hannah: Bye Amanda: Bye bye
---
Summary:
"""    


# eval_prompt ="""Summarize this dialog:
# Hannah: Hey, do you have Betty's number?
# Amanda: Lemme check
# Hannah: <file_gif>
# Amanda: Sorry, can't find it.
# Amanda: Ask Larry
# Amanda: He called her last time we were at the park together
# Hannah: I don't know him well
# Hannah: <file_gif>
# Amanda: Don't be shy, he's very nice
# Hannah: If you say so..
# Hannah: I'd rather you texted him
# Amanda: Just text him 🙂
# Hannah: Urgh.. Alright
# Hannah: Bye
# Amanda: Bye bye
# ---
# Summary:
# """    
# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# with torch.no_grad():
#     model_input["nat_atten_mask"] = nat_atten_mask

#     print(tokenizer.decode(remove_duplicate_tokens(model.generate(**model_input, max_new_tokens=100,upsampling_rate=upsampling_rate,\
#         use_cache=use_cache,special_decode=special_decode,more_step=more_step,output_scores=output_scores,return_dict_in_generate=return_dict_in_generate)[0]), skip_special_tokens=True))
#     print("Ans: \n Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry.")    
    
# eval_prompt = """
# Answer the below question:
# How to be a good student
# ---
# Answer:
# """    
# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# with torch.no_grad():
#     model_input["nat_atten_mask"] = nat_atten_mask

#     print(tokenizer.decode(remove_duplicate_tokens(model.generate(**model_input, max_new_tokens=100,upsampling_rate=upsampling_rate,\
#         use_cache=use_cache,special_decode=special_decode,more_step=more_step,output_scores=output_scores,return_dict_in_generate=return_dict_in_generate)[0]), skip_special_tokens=True))      
#     model =LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
#     print(tokenizer.decode(remove_duplicate_tokens(model.generate(**model_input, max_new_tokens=100,upsampling_rate=upsampling_rate,\
#         use_cache=use_cache,special_decode=special_decode,more_step=more_step,output_scores=output_scores,return_dict_in_generate=return_dict_in_generate)[0]), skip_special_tokens=True))
    
    

# eval_prompt = """
# Answer the below question:
# Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what’s your current position? Where is the person you just overtook?    
# ---
# Answer:
# """    
# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# with torch.no_grad():
#     model_input["nat_atten_mask"] = nat_atten_mask

#     print(tokenizer.decode(remove_duplicate_tokens(model.generate(**model_input, max_new_tokens=100,upsampling_rate=upsampling_rate,\
#         use_cache=use_cache,special_decode=special_decode,more_step=more_step,output_scores=output_scores,return_dict_in_generate=return_dict_in_generate)[0]), skip_special_tokens=True))