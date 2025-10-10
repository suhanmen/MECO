import os
import json
import traceback
from sklearn.metrics import accuracy_score, f1_score
import sys
import ast
import re
import pdb
import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
'''
python sh_weight_consensus.py java step2,3 multi-codecomplex-expertise Fasle regen
'''

def label_format(label):
    if label == 0:
        return "constant"
    elif label == 1:
        return "logn"
    elif label == 2:
        return "linear"
    elif label == 3:
        return "nlogn"
    elif label == 4:
        return "quadratic"
    elif label == 5:
        return "cubic"
    elif label == 6:
        return "exponential"
    else:
        return "error"

def back_name_format(model_name):
    if model_name == "deepseek-coder-7b-instruct-v1.5" or model_name == "deepseek-coder-7b-instruct-v1.5-codecomplex-simple":
        return "deepseek"
    elif model_name == "Meta-Llama-3.1-8B-Instruct" or model_name == "Meta-Llama-3.1-8B-Instruct-codecomplex-simple":
        return "llama"
    elif model_name == "Qwen2.5-Coder-7B-Instruct" or model_name == "Qwen2.5-Coder-7B-Instruct-codecomplex-simple":
        return "qwen"
    elif model_name == "Ministral-8B-Instruct-2410" or model_name == "Ministral-8B-Instruct-2410-codecomplex-simple":
        return "ministral"
    elif model_name == "DeepSeek-R1-Distill-Qwen-7B" or model_name == "DeepSeek-R1-Distill-Qwen-7B-codecomplex-simple":
        return "deepseekr1-qwen"
    elif model_name == "DeepSeek-R1-Distill-Llama-8B" or model_name == "DeepSeek-R1-Distill-Llama-8B-codecomplex-simple":
        return "deepseekr1-llama"
    else:
        return "error"

# setting
language_option = sys.argv[1]
data_option = sys.argv[2]
option_task = sys.argv[3]
add_expertise_classification_score = sys.argv[4]
expertise_regeneration = sys.argv[5]

# logit score path
if option_task == "single-no-expertise_multi-no-expertise-with-logit":
    option_logit = "single-no-expertise_multi-no-expertise"
elif option_task == "multi-codecomplex-expertise-with-logit":
    option_logit = "multi-codecomplex-expertise"
elif option_task in "single-no-expertise_multi-expertise-with-logit":
    option_logit = "single-no-expertise_multi-expertise"
elif option_task in "multi-codecomplex-expertise-with-logit":
    option_logit = "multi-codecomplex-expertise"
elif option_task in "single-expertise_multi-no-expertise-with-logit":
    option_logit = "single-expertise_multi-no-expertise"
else:
    option_logit = option_task
    
logit_score_path = []
logit_score_root = f"./scripts/sh_response/model/{option_logit}/{data_option}/{language_option}"
logit_score_name = os.listdir(logit_score_root)


# log setting
if expertise_regeneration == "no_regen":
    log_dir = f"./scripts/result/0.voting/pipeline/{option_task}/{data_option}/"
elif expertise_regeneration == "regen":
    log_dir = f"./scripts/result/0.voting/regen/pipeline/{option_task}/{data_option}/"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d")
log_filepath = os.path.join(log_dir, f"{option_task}_{language_option}_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,  # log level
    format="%(message)s",  # log format
    handlers=[
        logging.FileHandler(log_filepath, mode="w",encoding="utf-8"),  # save to file (terminal X)
    ]
)



# load logit file
if len(logit_score_name) == 7:
    logging.info("Expertise style : This is expertise model\n")
elif len(logit_score_name) == 4:
    logging.info("Expertise style :This is not expertise model\n")

for logit_name in logit_score_name:
    if len(logit_score_name) == 7:
        logit_model_name = os.listdir(os.path.join(logit_score_root, logit_name))
        if expertise_regeneration == "regen":
            logit_score_path.append(f"{logit_score_root}/{logit_name}/{logit_model_name[0]}/regen")
        else:
            logit_score_path.append(f"{logit_score_root}/{logit_name}/{logit_model_name[0]}")
    elif len(logit_score_name) == 4:
        logit_score_path.append(f"{logit_score_root}/{logit_name}")

total_logit_data = {}
logging.info(f"Logit scroe format :")
if len(logit_score_name) == 7:
    for load_path in logit_score_path:
        with open(f"{load_path}/logit_score.json", "r", encoding="utf-8") as f:
            
            if expertise_regeneration == "regen":
                if load_path.split('/')[-3] in ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential']:
                    name_format = f"{load_path.split('/')[-3]}-{back_name_format(load_path.split('/')[-2])}"
                    total_logit_data[name_format] = json.load(f)
                    logging.info(f"    {name_format} : {len(total_logit_data[name_format].keys())}")
                else:
                    pdb.set_trace()
            
            elif expertise_regeneration == "no_regen":
                if load_path.split('/')[-2] in ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential']:
                    name_format = f"{load_path.split('/')[-2]}-{back_name_format(load_path.split('/')[-1])}"
                    total_logit_data[name_format] = json.load(f)
                    logging.info(f"    {name_format} : {len(total_logit_data[name_format].keys())}")
                else:
                    pdb.set_trace()
                    
elif len(logit_score_name) == 4:
    for load_path in logit_score_path:
        with open(f"{load_path}/logit_score.json", "r", encoding="utf-8") as f:
            total_logit_data[back_name_format(load_path.split('/')[-1])] = json.load(f)
            logging.info(f"    {load_path.split('/')[-1]} : {len(total_logit_data[back_name_format(load_path.split('/')[-1])].keys())}")    




# load prediction dataset
if expertise_regeneration == "regen":
    with open(f"./scripts/result/0.voting/regen/pipeline/{option_task}/{data_option}/output-{language_option}-voting.json","r",encoding="utf-8") as t:
        total_voting_data = json.load(t)
elif expertise_regeneration == "no_regen":
    with open(f"./scripts/result/0.voting/pipeline/{option_task}/{data_option}/output-{language_option}-voting.json","r",encoding="utf-8") as t:
        total_voting_data = json.load(t)
    

total_output = []
label_list = []

# load label data
for idx in range(490):
    label_key = list(total_voting_data[str(idx)].keys())
    if label_key[-1] != "vote":
        label_list.append(total_voting_data[str(idx)][label_key[-1]]['true_complexity'])
    else:
        logging.info("label key error")
        pdb.set_trace()

# output voting data
for idx in range(490):    
    instance_score = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0} 
    
    for vote_key in total_voting_data[str(idx)].keys():
        if vote_key == "vote":
            continue 
        voting_idx = total_voting_data[str(idx)][vote_key]["predicted_complexity"]
        
        if voting_idx == -1:
            continue
        
        try:
            voting_score = total_logit_data[vote_key][str(idx)]['logit_score']
        except:
            if (label_format(total_voting_data[str(idx)][vote_key]["predicted_complexity"]) == vote_key.split('-')[0] and
                len(logit_score_name) == 7):
                voting_score = 1
            else:
                voting_score = 0
                
        #logging.info(f"{voting_idx} - {voting_score}")
        if add_expertise_classification_score == "True":
            # need to be written
            pass
        
        
        elif add_expertise_classification_score == "False":
            instance_score[voting_idx] += voting_score
    
    # output value
    max_keys = max(instance_score, key=instance_score.get)
    if instance_score[max_keys] == 0:
        total_output.append(-1) 
        #pdb.set_trace()
    else:
        total_output.append(max_keys)

# accuracy and f1 score calculation (f1 is calculated by average="macro" for multi-class)       
accuracy = accuracy_score(label_list, total_output)
f1_macro = f1_score(label_list, total_output, average="macro")
f1_weighted = f1_score(label_list, total_output, average='weighted')


# confusion matrix
valid_labels = [0, 1, 2, 3, 4, 5, 6, -1]  
cm = confusion_matrix(label_list, total_output, labels=valid_labels)
labels_display = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "exponential", "error"]
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels_display, yticklabels=labels_display)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix with Unknown')
plt.savefig(f"{log_dir}/{language_option}-weight_consensus-confusion_matrix.png", dpi=300, bbox_inches='tight')


logging.info(
    f"\n\nformat:\n"
    f"    Task : {option_task}\n"
    f"    Data : {data_option}\n"
    f"    PL-language : {language_option}\n"
    f"    add class score : {add_expertise_classification_score}\n"
    f"    Regeneration type : {expertise_regeneration}\n"
    f"    Ready data - Label : {len(label_list)} , Prediction : {len(total_output)}\n\n"
    f"Total Score(acc, f1):\n"
    f"    Accuracy : {accuracy}\n"
    f"    F1 Score(macro) : {f1_macro}\n"
    f"    F1 Score(weighted) : {f1_weighted}\n"
    
)


print(f"{option_task} :: {language_option} :: Done!!!")
