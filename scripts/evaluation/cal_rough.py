
from rouge_score import rouge_scorer

import os
import csv
# Initialize the lists for the target and hypothesis sentences
targets = []
hypotheses = []

root_path="/work/valex1377/semi_at_llama/llama_model/models_hf/"

experiments = ['KD_opentext_gen_80k_20k_ft_b3_WiKLDiv_WiSeqloss_WoChat_WoNoise']

datasets = ['samsum_dataset']
types = ['no_insert_waiting','insert_waiting','add_diff_para','insert_waiting_add_diff_para']
read_flies = ['generate-test.txt']
write_file = 'Rough.csv'


# Open the file



# Initialize a Rouge scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


for experiment in experiments :
    for dataset in datasets :
        for type in types :
            for read_file in read_flies :
                
                read_flie_path=os.path.join(root_path,experiment,dataset,type,read_file)
                # Check if the file exists
                if not os.path.isfile(read_flie_path):
                    # If the file does not exist, write '-' to data
                    data = [
                        ['Experiment', experiment],
                        ['Type', type],
                        ['Metric', 'Value'],
                        ['Rouge1 F-Measure', '--'],
                        ['Rouge2 F-Measure', '--'],
                        ['RougeL F-Measure', '--'],
                        ['Average N', '--'],
                        ['Average I', '--'],
                        ['Average Speed', '--']
                    ]                    
                else:
                    # read_flie_path=os.path.join(root_path,text_path)
                    # Initialize variables
                    target = ""
                    hypothesis = ""
                    all_targets = []
                    all_hypotheses = []
                    n_values = []
                    i_values = []
                    ratios = []
                    # Read the file
                    with open(read_flie_path, 'r') as file:
                        for line in file:
                            if line.startswith('T-'):
                                target = line[2:].strip()
                            elif line.startswith('H-'):
                                hypothesis += line[2:].strip() + " "
                            elif line.startswith('E-') or line.startswith('I-') or line.startswith('N-'):
                                # Add the target and hypothesis to the lists
                                all_targets.append(target)
                                all_hypotheses.append(hypothesis.strip())
                                hypothesis = ""
                            if line.startswith('N-'):
                                n_values.append(float(line[2:].strip()))
                            elif line.startswith('I-'):
                                i_values.append(float(line[2:].strip()))
                                if n_values:
                                    # Calculate the ratio of the last "N-" value to this "I-" value
                                    ratios.append(n_values[-1] / i_values[-1])

                    # Calculate Rouge score for all groups of sentences
                    all_targets_text = " ".join(all_targets)
                    all_hypotheses_text = " ".join(all_hypotheses)
                    scores = scorer.score(all_targets_text, all_hypotheses_text)

                    # Calculate the average of all "N-" and "I-" values
                    average_n = sum(n_values) / len(n_values) if n_values else 0
                    average_i = sum(i_values) / len(i_values) if i_values else 0
                    average_speed = (average_n/average_i)*100 

                    # Print the Rouge scores and averages
                    print("=====================================================================")
                    print("Rouge Scores:", scores)
                    print("Average N:", average_n)
                    print("Average I:", average_i)
                    print("Average Speed:", average_speed)
                    
                    write_file_path=os.path.join(root_path,experiment,dataset,type,write_file)
                    # Prepare data for CSV
                    data = [
                        ['Experiment', experiment],
                        ['Type', type],                        
                        ['Metric', 'Value'],
                        ['Rouge1 F-Measure', str(round(scores['rouge1'][0]*100,2))+'/'+str(round(scores['rouge1'][1]*100,2))+'/'+str(round(scores['rouge1'][2]*100,2))],
                        ['Rouge2 F-Measure', str(round(scores['rouge2'][0]*100,2))+'/'+str(round(scores['rouge2'][1]*100,2))+'/'+str(round(scores['rouge2'][2]*100,2))],
                        ['RougeL F-Measure', str(round(scores['rougeL'][0]*100,2))+'/'+str(round(scores['rougeL'][1]*100,2))+'/'+str(round(scores['rougeL'][2]*100,2))],
                        ['Average N', average_n],
                        ['Average I', average_i],
                        ['Average Speed', average_speed]
                    ]
                    os.makedirs(os.path.dirname(write_file_path), exist_ok=True)
                    # Write data to CSV file
                    with open(write_file_path, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(data)
            
            # Write all data to CSV file
            write_file_path=os.path.join(root_path,'all_result.csv')
            with open(write_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)