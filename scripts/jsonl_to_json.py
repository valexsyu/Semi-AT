        
import json

data = []
with open('/work/valex1377/semi_at_llama/kd_datasets/mt_bench/question.jsonl', 'r') as jsonl_file:
    for line in jsonl_file:
        json_obj = json.loads(line)
        if len(json_obj["turns"]) > 1 :
            breakpoint()
            print("multi turns========")
        json_obj["turns"] = json_obj["turns"][0] if json_obj["turns"] else ""
        data.append(json_obj)


with open('/work/valex1377/semi_at_llama/kd_datasets/mt_bench/question.json', 'w') as json_file:
    json.dump(data, json_file)



