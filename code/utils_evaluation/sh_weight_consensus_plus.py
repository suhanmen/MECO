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
from collections import Counter
'''
python sh_weight_consensus.py java step2,3 multi-codecomplex-expertise Fasle regen
'''

def step1_score_format(language_option):
    data_path = "../code/output_scoring/pipeline/MECO/single/step1-10%"
    all_data_name = os.listdir(data_path)
    step1_score = {language_option : {}}

    for data_name in [all_data for all_data in all_data_name if all_data in ['deepseek', 'llama', 'ministral', 'qwen', 'deepseekr1-qwen', 'deeseekr1-llama']]:
        step1_path = f"{data_path}/{data_name}/{data_name}-{language_option}-expertise.json"
        step1_score[language_option][data_name] = json.load(open(step1_path, "r", encoding="utf-8"))
    return step1_score


def step1_json(language_option):
    data_path = "../code/output_scoring/pipeline/MECO/single/step1-10%"
    all_data_name = os.listdir(data_path)
    step1_class = {language_option : {}}
    filter_list = {language_option : {}}
    
    for data_name in [all_data for all_data in all_data_name if all_data in ['deepseek', 'llama', 'ministral', 'qwen', 'deepseekr1-qwen', 'deeseekr1-llama']]:
        step1_path = f"{data_path}/{data_name}/{data_name}-{language_option}-combined_output.jsonl"
        with open(step1_path , "r", encoding="utf-8") as f:
            step1_class[language_option][data_name] = [json.loads(line) for line in f]
    
    for i in step1_class[language_option].keys():
        filter_list[language_option][i] = {}
        for idx, data_name in enumerate(step1_class[language_option][i]):    
            filter_list[language_option][i][idx] = {'true_complexity':data_name['true_complexity'], 'predicted_complexity':data_name['predicted_complexity']}
    
    
    # check label distribution
    label_list = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '-1':0}
    sample = 'deepseek'
    for idx in filter_list[language_option][sample].keys():
        label_list[str(filter_list[language_option][sample][idx]['true_complexity'])] += 1
    #print(f"Count of label : {label_list}\n")
    
    predict_list = {}
    class_count = {}
    for data_name in filter_list[language_option].keys():
        predict_list[data_name] = []
        class_count[data_name] = {}
        for idx in filter_list[language_option][data_name].keys():
            predict_list[data_name].append(filter_list[language_option][data_name][idx]['predicted_complexity'])
        #print(f"{data_name}-predict_list : {sorted(Counter(predict_list[data_name]).items())}")
        
        for count in sorted(Counter(predict_list[data_name]).items()):    
            if count[0] == -1:
                continue
            class_count[data_name][count[0]] = count[1] / label_list.get(str(0))
            
                
    #print("\nLabel format : 0-constant, 1-logn, 2-linear, 3-nlogn, 4-quadratic, 5-cubic, 6-exponential")
    #for class_c in class_count.keys():
    #    print(f"{class_c}-class_count : {class_count[class_c]}")

    
    #print("\n")
    normalize_class = {}
    for data_name in class_count.keys():
        total_score =0
        normalize_class[data_name] = {}
        for class_idx, class_score in class_count[data_name].items():
            total_score += class_score
            normalize_class[data_name][class_idx] = class_score
        for label_idx in normalize_class[data_name].keys():
            normalize_class[data_name][label_idx] =normalize_class[data_name][label_idx]/total_score
        
        
        if len(normalize_class[data_name].keys()) == 7:
            pass
        else:
            search_list = [0, 1, 2, 3, 4, 5, 6]
            for key in normalize_class[data_name].keys():
                search_list.remove(key)
            for i in search_list:
                normalize_class[data_name][i] = 0
        
        
    #for data_name in normalize_class.keys():
    #    print(f"{data_name}-Normalize class : {normalize_class[data_name]}")
    
    #pdb.set_trace()
    return normalize_class




def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x))  # Stabilized softmax
    return exp_x / exp_x.sum()



def normalize_logits(logit_dict):
    """Normalize logit scores per instance to ensure consistency."""
    for idx in logit_dict.keys():
        logits = np.array([logit_dict[idx][exp]['logit_score'] for exp in logit_dict[idx]])
        logit_sum = logits.sum()
        if logit_sum > 0:
            for exp in logit_dict[idx]:
                logit_dict[idx][exp]['normalized_score'] = logit_dict[idx][exp]['logit_score'] / logit_sum
        else:
            for exp in logit_dict[idx]:
                logit_dict[idx][exp]['normalized_score'] = 0  # No valid logit

def compute_weighted_consensus(total_voting_data, total_logit_data, step1_class, step1_score):
    """Compute weighted consensus based on expertise and logit confidence."""
    total_output = []

    # output voting data
    for idx in range(490):    
        instance_score = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0} 
        total_weight = 0
        
        for vote_key in total_voting_data[str(idx)].keys():
            if vote_key == "vote":
                continue 
            predicted_class = total_voting_data[str(idx)][vote_key]["predicted_complexity"]
            
            if predicted_class == -1:
                continue
            
            try:
                logit_score = total_logit_data[vote_key][str(idx)]['logit_score']
                
                
                # logit is 0 but pipeline predicted correctly
                if logit_score == 0 and label_format(total_voting_data[str(idx)][vote_key]["predicted_complexity"]) == vote_key.split('-')[0]:
                    logit_score = 1
                elif logit_score == 0 and label_format(total_voting_data[str(idx)][vote_key]["predicted_complexity"]) != vote_key.split('-')[0]:
                    logit_score = 0.5
           
            except:
                if (label_format(total_voting_data[str(idx)][vote_key]["predicted_complexity"]) == vote_key.split('-')[0] and
                    len(logit_score_name) == 7):
                    logit_score = 1
                else:
                    logit_score = 0
                    
            if vote_key.split('-')[0] == label_format(predicted_class):
                weight = 2 # Alpha
            else:
                weight = 0.5 # Beta
            
            #confidence_weight = softmax([logit_score])[0]
            confidence_weight = logit_score
            
            # if  total_voting_data[str(idx)][vote_key]["predicted_complexity"] != -1:                
                # try:
                #     append_class_score = step1_class[vote_key.split('-')[1]][total_voting_data[str(idx)][vote_key]["predicted_complexity"]]
                # except:
                #     pdb.set_trace()
            
            final_weight = weight * confidence_weight #+ append_class_score
            instance_score[predicted_class] += final_weight

        '''
        best
        weight=1 , weight = 0.85
        confidence_weight = logit_score
        java acc : 60, f1 : 50.50
        python acc : 57.14, f1:46.15
        error occurs : -1
        
        '''

        
        # output value
        max_keys = max(instance_score, key=instance_score.get)
        if instance_score[max_keys] == 0:
            #print(total_voting_data[str(idx)]['vote'])
            #print(instance_score)
            #pdb.set_trace()
            total_output.append(-1) 
        else:
            total_output.append(max_keys)

    return total_output


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
expertise_regeneration = sys.argv[4]

logit_options  = ["single-no-expertise_multi-no-expertise", "single-no-expertise_multi-expertise", "single-expertise_multi-no-expertise",
			   "multi-codecomplex-expertise", "multi-codecomplex-expertise-with-confidence", "multi-codecomplex-expertise-with-confidence-2"]


step1_score = step1_score_format(language_option)
step1_class = step1_json(language_option)


# logit score path
if option_task in "single-no-expertise_multi-no-expertise-with-logit":
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
    option_logit = "multi-codecomplex-expertise"


logit_score_path = []
logit_score_root = f"../output_initialize/model/MECO/{option_logit}/{data_option}/{language_option}"
logit_score_name = os.listdir(logit_score_root)


# log setting
if expertise_regeneration == "no_regen":
    log_dir = f"./scripts/result/0.voting/pipeline/{option_task}/{data_option}/"
elif expertise_regeneration == "regen":
    log_dir = f"./scripts/result/0.voting/regen/pipeline/{option_task}/{data_option}/"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d")
log_filepath = os.path.join(log_dir, f"{option_task}_{language_option}_{timestamp}_plus.log")
logging.basicConfig(
    level=logging.INFO,  # log level
    format="%(message)s",  # log format
    handlers=[
        logging.FileHandler(log_filepath, mode="w",encoding="utf-8"),  # save to file (terminal X)
    ]
)


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
    with open(f"../code/output_voting/MECO/regen/{option_task}/{data_option}/output-{language_option}-voting.json","r",encoding="utf-8") as t:
        total_voting_data = json.load(t)
elif expertise_regeneration == "no_regen":
    with open(f"../code/output_voting/MECO/{option_task}/{data_option}/output-{language_option}-voting.json","r",encoding="utf-8") as t:
        total_voting_data = json.load(t)

label_list = []

# load label data
for idx in range(490):
    label_key = list(total_voting_data[str(idx)].keys())
    if label_key[-1] != "vote":
        label_list.append(total_voting_data[str(idx)][label_key[-1]]['true_complexity'])
    else:
        logging.info("label key error")


normalize_logits(total_logit_data)
total_output = compute_weighted_consensus(total_voting_data, total_logit_data, step1_class, step1_score)

# accuracy and f1 score calculation (f1 is calculated by average="macro" for multi-class)       
accuracy = accuracy_score(label_list, total_output)
f1_macro = f1_score(label_list, total_output, average="macro")
f1_weighted = f1_score(label_list, total_output, average="weighted")


# confusion matrix
# valid_labels = [0, 1, 2, 3, 4, 5, 6, -1]  
# cm = confusion_matrix(label_list, total_output, labels=valid_labels)
# labels_display = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "exponential", "error"]
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels_display, yticklabels=labels_display)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix with Unknown')
# plt.savefig(f"{log_dir}/{language_option}-weight_consensus_plus-confusion_matrix.png", dpi=300, bbox_inches='tight')

valid_labels = [0, 1, 2, 3, 4, 5, 6, -1]
cm_temp = confusion_matrix(label_list, total_output, labels=valid_labels)
print(cm_temp)

encoded_all_labels = []
encoded_all_preds = []
labels = [
        r'$O(1)$',  # constant 0
        r'$O(\log n)$',  # logn 1
        r'$O(n)$',  # linear 2
        r'$O(n \log n)$',  # nlogn 3
        r'$O(n^2)$',  # quadratic 4
        r'$O(n^3)$',  # cubic 5
        r'$O(2^n)$',  # exponential 6
        ]

labels_ERROR = [
        r'$O(1)$',  # constant 0
        r'$O(\log n)$',  # logn 1
        r'$O(n)$',  # linear 2
        r'$O(n \log n)$',  # nlogn 3
        r'$O(n^2)$',  # quadratic 4
        r'$O(n^3)$',  # cubic 5
        r'$O(2^n)$',  # exponential 6
        'Error' # -1
        ]


class_complexity_dict = {0: r'$O(1)$', 1: r'$O(\log n)$', 2: r'$O(n)$', 3: r'$O(n \log n)$', 4: r'$O(n^2)$', 5: r'$O(n^3)$', 6: r'$O(2^n)$', -1:'Error'}
for l,p in zip(label_list, total_output):
    encoded_all_labels.append(class_complexity_dict[l])
    
    try:
        encoded_all_preds.append(class_complexity_dict[p])
    except:
        pdb.set_trace()
        
# confusion matrix
cm = confusion_matrix(encoded_all_labels, encoded_all_preds, labels=labels, normalize='true')
#labels_display = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "exponential", "Unknown"]




sns.set(font_scale=1.8)
labels_display = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "exponential"]
plt.figure(figsize=(12, 9))
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=labels_ERROR, yticklabels=labels)

ax.tick_params(axis='x', labelsize=18)  # Adjust x-axis label size
ax.tick_params(axis='y', labelsize=16)  # Adjust y-axis label size
plt.xlabel('Predictions', fontsize=20)
plt.ylabel('True labels', fontsize=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)  # Adjust colorbar label size

plt.xlabel('Predictions')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of MEC³O', fontsize=20)
plt.savefig(f"{log_dir}/{language_option}_MECO_confusion.png", dpi=300, bbox_inches='tight')




logging.info(
    f"\n\nformat:\n"
    f"    Task : {option_task}\n"
    f"    Data : {data_option}\n"
    f"    PL-language : {language_option}\n"
    f"    add class score : True\n"
    f"    Regeneration type : {expertise_regeneration}\n"
    f"    Ready data - Label : {len(label_list)} , Prediction : {len(total_output)}\n\n"
    f"Total Score(acc, F1 Macro, and F1 Weighted):\n"
    f"    Accuracy : {accuracy}\n"
    f"    F1 Macro Score : {f1_macro}\n"
    f"    F1 Weighted Score : {f1_weighted}\n"
)

print(f"Accuracy: {accuracy:.4f}, F1 Macro Score: {f1_macro:.4f}, F1 Weighted Score: {f1_weighted:.4f}")

if language_option == "python":
    print('\n\n')
