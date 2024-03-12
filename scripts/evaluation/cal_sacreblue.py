
# import sacrebleu
# import os
# # Initialize the lists for the target and hypothesis sentences
# targets = []
# hypotheses = []

# root_path="/work/valex1377/semi_at_llama/llama_model/models_hf/"
# # text_path="KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e/iwslt2017deennat_dataset/add_diff_para/generate-test.txt"
# # text_path="KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e/iwslt2017deennat_dataset/insert_waiting/generate-test.txt"
# # text_path="KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_1e/iwslt2017deennat_dataset/add_diff_para_insert_waiting/generate-test.txt"
# # text_path="KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_wochat_2e/iwslt2017deennat_dataset/add_diff_para/generate-test.txt"
# text_path="KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_wochat_2e/iwslt2017deennat_dataset/insert_waiting/generate-test.txt"
# # text_path="KD_opentext_gen_20k_ft_b3_WoKLDiv_WiSeqloss_wochat_2e/iwslt2017deennat_dataset/add_diff_para_insert_waiting/generate-test.txt"
# # text_path="LlamaSemiATForCausalLM/iwslt2017deennat_dataset/insert_waiting/generate-test.txt"
# # text_path="LlamaSemiATForCausalLM_wochat/iwslt2017deennat_dataset/insert_waiting/generate-test.txt"
# # text_path="LlamaSemiATForCausalLM_wochat/iwslt2017deennat_dataset/insert_waiting/generate-test.txt"

# file_path=os.path.join(root_path,text_path)
# # Open the file
# with open(file_path, 'r') as file:
#     # Read the lines
#     lines = file.readlines()

#     # Initialize the current target and hypothesis
#     current_target = None
#     current_hypothesis = None
#     n_values = []
#     i_values = []
#     ratios = []
    
#     # Iterate over the lines
#     for line in lines:
#         # Split the line into the tag and the sentence
#         parts = line.strip().split('-', 1)

#         # Check if the line contains a '-'
#         if len(parts) == 2:
#             tag, sentence = parts

#             # Remove the tab character from the sentence
#             sentence = sentence.strip('\t')

#             # Add the sentence to the appropriate list
#             if tag == 'T':
#                 current_target = sentence
#             elif tag == 'H':
#                 current_hypothesis = sentence
            
                
                
#                 # Add the current target and hypothesis to the lists
#                 if current_target and current_hypothesis:
#                     targets.append([current_target])
                    
#                     hypotheses.append(current_hypothesis)
                    
#                     # Reset the current target and hypothesis
#                     current_target = None
#                     current_hypothesis = None
                    
#             if line.startswith('N-'):
#                 n_values.append(float(line[2:].strip()))
#             elif line.startswith('I-'):
#                 i_values.append(float(line[2:].strip()))
#                 if n_values:
#                     # Calculate the ratio of the last "N-" value to this "I-" value
#                     ratios.append(n_values[-1] / i_values[-1])                    

# # Calculate the c

# bleu = sacrebleu.corpus_bleu(hypotheses, targets)
# # print("Corpus BLEU score:", bleu.score)
# # Calculate the average of all "N-" and "I-" values
# average_n = sum(n_values) / len(n_values) if n_values else 0
# average_i = sum(i_values) / len(i_values) if i_values else 0
# average_speed = (average_n/average_i)*100 

# # Print the Rouge scores and averages
# print("Corpus BLEU score:", round(bleu.score,2))
# print("Average N:", round(average_n,2))
# print("Average I:", round(average_i,2))
# print("Average Speed:", round(average_speed-100,2))






import sacrebleu

import os
import csv
# Initialize the lists for the target and hypothesis sentences
targets = []
hypotheses = []

root_path="/work/valex1377/semi_at_llama/llama_model/models_hf/"





experiments = ['KD_opentext_gen_80k_20k_ft_b3_WoKLDiv_WiSeqloss_wochat','KD_opentext_gen_80k_40k_ft_b3_WoKLDiv_WiSeqloss_wochat',
             'KD_opentext_gen_80k_60k_ft_b3_WoKLDiv_WiSeqloss_wochat','KD_opentext_gen_80k_80k_ft_b3_WoKLDiv_WiSeqloss_wochat',
             'llama-2-7b', 'llama-2-7b-chat']

datasets = ['iwslt2017deennat_dataset']
types = ['no_insert_waiting','insert_waiting','add_diff_para','insert_waiting_add_diff_para']
read_flies = ['generate-test.txt']
write_file = 'BLEU.csv'




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
                        ['BLEU', '--'],
                        ['Average T', '--'],
                        ['Average N', '--'],
                        ['Average I', '--'],
                        ['Average Speed', '--']
                    ]                  
                else:
                    with open(read_flie_path, 'r') as file:
                        # Read the lines
                        lines = file.readlines()

                        # Initialize the current target and hypothesis
                        current_target = None
                        current_hypothesis = None
                        n_values = []
                        i_values = []
                        e_values = []
                        ratios = []
                        hypotheses = []
                        targets = []
                        # Iterate over the lines
                        for line in lines:
                            # Split the line into the tag and the sentence
                            parts = line.strip().split('-', 1)

                            # Check if the line contains a '-'
                            if len(parts) == 2:
                                tag, sentence = parts

                                # Remove the tab character from the sentence
                                sentence = sentence.strip('\t')

                                # Add the sentence to the appropriate list
                                if tag == 'T':
                                    current_target = sentence
                                elif tag == 'H':
                                    current_hypothesis = sentence
                                
                                    
                                    
                                    # Add the current target and hypothesis to the lists
                                    if current_target and current_hypothesis:
                                        targets.append([current_target])
                                        
                                        hypotheses.append(current_hypothesis)
                                        
                                        # Reset the current target and hypothesis
                                        current_target = None
                                        current_hypothesis = None
                                        
                                if line.startswith('N-'):
                                    n_values.append(float(line[2:].strip()))
                                elif line.startswith('I-'):
                                    i_values.append(float(line[2:].strip()))
                                    if n_values:
                                        # Calculate the ratio of the last "N-" value to this "I-" value
                                        ratios.append(n_values[-1] / i_values[-1])     
                                elif line.startswith('E-'):         
                                    e_values.append(float(line[2:].strip()))

                    # Calculate the c
                    bleu = sacrebleu.corpus_bleu(hypotheses, targets)
                    
                    # print("Corpus BLEU score:", bleu.score)
                    # Calculate the average of all "N-" and "I-" values
                    average_e = sum(e_values) / len(e_values) if e_values else 0
                    average_n = sum(n_values) / len(n_values) if n_values else 0
                    average_i = sum(i_values) / len(i_values) if i_values else breakpoint()
                    try:    
                        average_speed = (average_n/average_i)*100 
                    except:
                        breakpoint()

                    # Print the Rouge scores and averages
                    print("Corpus BLEU score:", round(bleu.score,5))
                    print("Average T:", round(average_e,2))
                    print("Average N:", round(average_n,2))
                    print("Average I:", round(average_i,2))
                    print("Average Speed:", round(average_speed-100,2))
                    
                    write_file_path=os.path.join(root_path,experiment,dataset,type,write_file)
                    # Prepare data for CSV
                    data = [
                        ['Experiment', experiment],
                        ['Type', type],                        
                        ['Metric', 'Value'],
                        ['BLEU', round(bleu.score,5)],
                        ['Average T', round(average_e,2)],
                        ['Average N', round(average_n,2)],
                        ['Average I', round(average_i,2)],
                        ['Average Speed', round(average_speed-100,2)]
                    ]
                    os.makedirs(os.path.dirname(write_file_path), exist_ok=True)
                    # Write data to CSV file
                    with open(write_file_path, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(data)
            
            write_file_path=os.path.join(root_path,'all_bleu_result.csv')
            with open(write_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)