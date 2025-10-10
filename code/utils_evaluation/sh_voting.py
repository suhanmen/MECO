import json
import os
import sys
import pdb
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def back_name_format(name):
    if name == "Qwen2.5":
        return "qwen"
    elif name == "Meta":
        return "llama"
    elif name == "Ministral":
        return "ministral"
    elif name == "DeepSeek":
        return "deepseek"
    elif name == "deepseek":
        return name
    elif name == "Mistral":
        return "mistral"
    elif name == "CodeLlama":
        return "codellama"
    elif name == "codegemma":
        return "codegemma"    
    else:
        pdb.set_trace()
        return "error"

## voting
## ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential']
complexity_list = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}


data_option = sys.argv[1]
option_task = sys.argv[2]
generation_type = sys.argv[3]
target_lang = sys.argv[4]
TAG = sys.argv[5]

try:
    expertise_regeneration = sys.argv[6]
except:
    expertise_regeneration = "no_regen"

if isinstance(target_lang, str) and ',' in target_lang:
    target_lang = target_lang.split(',')
elif isinstance(target_lang, str):
    target_lang = [target_lang]

not_expertise_result = [
                        "single", 
                        'single-fewshot', 
                        'single-no-expertise_multi-no-expertise', 
                        "single-expertise_multi-no-expertise",  
                        "single-expertise_multi-no-expertise-with-logit", 
                        'single-no-expertise_multi-no-expertise-with-logit', 
                        'judge_model-Ns_Nm', 'judge_model-Ys_Nm', 'judge_model-Ns_Ym', 'judge_model-Ys_Ym',
                        'REC-initialize','REC-discussion1', 'REC-discussion2',
                        'CMD-Group1', 'CMD-Group2', 
                        "CMD-Group1_discussion1", "CMD-Group2_discussion1", 
                        "CMD-Group1_discussion2", "CMD-Group2_discussion2",
                        "CMD-tie",
                        "MAD-affirmative_discussion1", "MAD-negative_discussion1",
                        "MAD-affirmative_discussion2", "MAD-negative_discussion2",
                        "MAD-affirmative_discussion3", "MAD-negative_discussion3",
                        "MAD-affirmative_discussion4", "MAD-negative_discussion4",
                        "MAD-affirmative_discussion5", "MAD-negative_discussion5",
                        "MAD-judge1", "MAD-judge2", "MAD-judge3", "MAD-judge4", "MAD-judge5",
                        ]


########################################################################################
# load result data
languages = target_lang
if expertise_regeneration == "no_regen":
    base_path = "../code/output_scoring"
else:
    base_path = f"../code/output_scoring/regen"
    
if data_option == "step2,3" and option_task not in  not_expertise_result:
    model_path = [f"{base_path}/{generation_type}/{TAG}/{option_task}/{data_option}/{language}/{expertise}" for expertise in ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential'] for language in languages]
    existing_files = [f"{path}/{f}" for path in model_path for f in os.listdir(path) if f.endswith(".jsonl")]

elif option_task in not_expertise_result:
    model_dir = f"{base_path}/{generation_type}/{TAG}/{option_task}/{data_option}"
    if os.path.exists(model_dir):
        model_names = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        model_path = [os.path.join(model_dir, model_name) for model_name in model_names]
        existing_files = [f for path in model_path for f in os.listdir(path) if f.endswith(".jsonl")]

java_data = {}
python_data = {}
for data_path in existing_files:
    if data_option == "step2,3" and option_task not in not_expertise_result:
        if "java" in data_path.split('/')[-1]: 
            with open(data_path, "r") as f:
                java_data[f"{data_path.split('/')[-2]}-{back_name_format(data_path.split('/')[-1].split('-')[0])}"] =  [json.loads(line) for line in f]
        
        elif "python" in data_path.split('/')[-1]:
            with open(data_path, "r") as f:
                python_data[f"{data_path.split('/')[-2]}-{back_name_format(data_path.split('/')[-1].split('-')[0])}"] =  [json.loads(line) for line in f]
    
    
    else:
        model_name = data_path.split("-")[0]
        if expertise_regeneration == "no_regen":
            main_path = f"../code/output_scoring/{generation_type}/{TAG}/{option_task}/{data_option}/{model_name}/"
        elif expertise_regeneration == "regen":
            main_path = f"../code/output_scoring/regen/{generation_type}/{TAG}/{option_task}/{data_option}/{model_name}/"
    
        if "java" in data_path: 
            with open(main_path + data_path, "r") as f:
                java_data[model_name] = [json.loads(line) for line in f]
        
        elif "python" in data_path :
            with open(main_path + data_path, "r") as f:
                python_data[model_name] = [json.loads(line) for line in f]

languages_data = []
for lang in languages:
    if lang == "java":
        languages_data.append(java_data)
    elif lang == "python":
        languages_data.append(python_data)
########################################################################################
## voting
for pl_name, full_data in enumerate(languages_data):
    if pl_name ==0:
       pl_name = "java"
    elif pl_name ==1:
       pl_name = "python"
    
    label_list = []
    total_predict = []
    
    output = {}
    vote_count = 0
    zero_count = 0
    max_correct_vote_count = 0
    min_correct_vote_count = 0
    correct_multi_vote_count = 0
    
    
    try:
        model_key = list(full_data.keys())[0]
    except:
        pdb.set_trace()
        
    number_of_data = len(full_data[model_key])
       
    for i in range(number_of_data):
        output[i] = {"vote" : [0,0,0,0,0,0,0]}

    for model in full_data.keys():
        for idx,example in enumerate(full_data[model]):
            if example["predicted_complexity"] != -1:
                output[idx]["vote"][example["predicted_complexity"]] += 1
            output[idx][model] = {key: example[key] for key in example.keys() if key not in ['generated_text', 'original_data']}

            
            
    for check in range(len(output)):
        if data_option == "step2,3":
            if max(output[check]["vote"]) == 7:
                vote_count += 1
            if max(output[check]["vote"]) == 0:
                zero_count += 1

        else:
            if max(output[check]["vote"]) == 4:
                vote_count += 1
            if max(output[check]["vote"]) == 0:
                zero_count += 1
        
        max_value = max(output[check]["vote"])
        max_indices = [i for i, v in enumerate(output[check]["vote"]) if v == max_value]
        
        if max_value == 0:
            pred = -1           
        else:
            pred = max_indices[0] 
        total_predict.append(pred)
        
        label_list.append(output[check][model_key]["true_complexity"])
        
        if output[check]["vote"].count(max_value) > 1:
            pass
        else:
            max_vote_predict = output[check]["vote"].index(max_value)
            label = output[check][model_key]["true_complexity"]
            
            if max_vote_predict == label:
                max_correct_vote_count += 1
        
        min_value = min(output[check]["vote"])
        if output[check]["vote"].count(min_value) > 1:
            pass
        else:
            min_vote_predict = output[check]["vote"].index(min_value)
            label = output[check][model_key]["true_complexity"]
            if min_vote_predict == label:
                min_correct_vote_count += 1 
                
        if len(max_indices) > 1:
            label = output[check][model_key]["true_complexity"]
            
            if label == max_indices[0]: # If there are multiple max values, the first one is selected
                correct_multi_vote_count += 1
    
    
    # confusion matrix
    # valid_labels = [0, 1, 2, 3, 4, 5, 6, -1]  
    # cm = confusion_matrix(label_list, total_predict, labels=valid_labels)
    # labels_display = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "exponential", "error"]
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels_display, yticklabels=labels_display)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix with Unknown')     

    total_output = total_predict
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
            'ERROR'  # error -1
            ]


    class_complexity_dict = {0: r'$O(1)$', 1: r'$O(\log n)$', 2: r'$O(n)$', 3: r'$O(n \log n)$', 4: r'$O(n^2)$', 5: r'$O(n^3)$', 6: r'$O(2^n)$', -1:'ERROR'}
    for l,p in zip(label_list, total_output):
        encoded_all_labels.append(class_complexity_dict[l])
        encoded_all_preds.append(class_complexity_dict[p])

    cm = confusion_matrix(encoded_all_labels, encoded_all_preds, labels=labels, normalize='true')




    sns.set(font_scale=1.8)
    labels_display = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "exponential"]
    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=labels, yticklabels=labels)


    ax.tick_params(axis='x', labelsize=18)  # Adjust x-axis label size
    ax.tick_params(axis='y', labelsize=16)  # Adjust y-axis label size
    plt.xlabel('Predictions', fontsize=20)
    plt.ylabel('True labels', fontsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)  # Adjust colorbar label size

    plt.xlabel('Predictions')
    plt.ylabel('True Labels')
    if option_task == "single-no-expertise_multi-no-expertise":
        plt.title('Confusion Matrix of Debate with Majority Vote', fontsize=20)
    else:
        plt.title('Confusion Matrix of MEC³O', fontsize=20)
    #plt.savefig(f"{log_dir}/{language}_MECO_confusion.png", dpi=300, bbox_inches='tight')
    
########################################################################################
## print result
    print(f"{pl_name}-same vote sample count (all same vote) = ", vote_count) 
    print(f"{pl_name}-all predict failed vote sample count (all failed) = ", zero_count) 
    print(f"{pl_name}-correct vote count (max) (correct majority vote) = ", max_correct_vote_count)
    print(f"{pl_name}-correct vote count (min) (correct minority vote) = ", min_correct_vote_count)
    print(f"{pl_name}-correct vote count (multi) (correct tie vote) = ", correct_multi_vote_count, "\n")
    
    print(f"Base Majority Vote (Accuracy, F1-score)")
    print(f"Total number of data: {number_of_data}")
    print(f"Accuracy: {accuracy_score(label_list, total_predict)}")
    print(f"F1-score(macro): {f1_score(label_list, total_predict, average='macro')}")
    print(f"F1-score(weighted): {f1_score(label_list, total_predict, average='weighted')}\n\n")
    
########################################################################################
## save result
    if expertise_regeneration == "no_regen":
        base_output_path=f"../code/output_voting"
    elif expertise_regeneration == "regen":
        base_output_path=f"../code/output_voting/regen"
    
    with open(f"{base_output_path}/{generation_type}/{TAG}/{option_task}/{data_option}/output-{pl_name}-voting.json", "w") as f:
        f.write(json.dumps(output, indent=4))
        plt.savefig(f"{base_output_path}/{generation_type}/{TAG}/{option_task}/{data_option}/{pl_name}-voting-confusion_matrix.png", dpi=300, bbox_inches='tight')