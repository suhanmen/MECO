import os
import json
import traceback 
from sklearn.metrics import accuracy_score, f1_score
import sys
import ast
import re
import pdb
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

'''
python sh_scoring.py qwen python single full
python sh_scoring.py qwen python single sample_pro-1050
'''

expertise_regeneration = 'no_regen'
not_expertise_result = [
                        "single", 
                        'single-fewshot', 
                        'single-no-expertise_multi-no-expertise', 
                        "single-expertise_multi-no-expertise",  
                        "single-expertise_multi-no-expertise-with-logit", 
                        'single-no-expertise_multi-no-expertise-with-logit', 
                        'judge_model-Ns_Nm', 'judge_model-Ys_Nm', 'judge_model-Ns_Ym', 'judge_model-Ys_Ym',
                        'REC-initialize', 'REC-discussion1', 'REC-discussion2',
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


def ensure_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def back_name_format(model_name):
    if model_name == "deepseek-coder-7b-instruct-v1.5-codecomplex-simple":
        return "deepseek"
    elif model_name == "Meta-Llama-3.1-8B-Instruct-codecomplex-simple":
        return "llama"
    elif model_name == 'Qwen2.5-Coder-7B-Instruct-codecomplex-simple':
        return "qwen"
    elif model_name == 'Ministral-8B-Instruct-2410-codecomplex-simple' :
        return 'ministral'
    elif model_name == 'CodeLlama-7b-Instruct-hf-codecomplex-simple':
        return 'codellama'
    elif model_name == 'codegemma-7b-it-codecomplex-simple':
        return 'codegemma'
    elif model_name == 'Mistral-7B-Instruct-v0.3-codecomplex-simple':
        return 'mistral'
    elif model_name == "DeepSeek-R1-Distill-Qwen-7B-codecomplex-simple":
        return "deepseekr1-qwen"
    elif model_name == "DeepSeek-R1-Distill-Llama-8B-codecomplex-simple":
        return "deepseekr1-llama"
    else:
        return "error"

# 📌 JSON error correction function
def fix_json_format(json_string, model_name=None):
    """
    JSON string is broken, automatically correct the function.
    """
    if not json_string:
        return None

    json_string = json_string.strip()

    # check JSON start and end
    if not json_string.startswith("{") or not json_string.endswith("}"):
        print("⚠️ JSON is not properly wrapped. Need to be fixed.")
        return None

    if r"\*" in json_string:
        json_string = json_string.replace(r"\*", "\\\\*")  
    
    # remove control characters (newline, tab, etc.)
    json_string = re.sub(r'[\n\t]', ' ', json_string)

    # automatically wrap string value after "Explanation":
    json_string = re.sub(r'("explanation":)\s*([A-Za-z])', r'\1 "\2', json_string)

    # add last quote (if explanation value is not properly terminated)
    json_string = re.sub(r'("explanation": "[^"]+$)', r'\1"', json_string)

    # remove single quotes (JSON uses double quotes by default)
    json_string = json_string.replace("\\'", "'")

    return json_string

# 📌 complexity conversion function
def adjust_complexity_format(complexity):
    complexity = complexity.lower()
    symbol_mapping = {
        'o(1)': 'constant', 
        'o(log n)': 'logn',
        'o(n log n)': 'nlogn',
        'o(n)': 'linear', 
        'o(n^2)': 'quadratic', 
        'o(n^3)': 'cubic',
        
        'nlogn' : 'nlogn',
        'n log n' : 'nlogn',
        'n' : 'linear', 
        'n^2' : 'quadratic', 
        'n^3' : 'cubic',
        'np' : 'exponential', 
        'factorial' : 'exponential',
        'logarithmic' : 'logn',
    }
    
    if complexity in complexity_list:
        return complexity
    return symbol_mapping.get(complexity, -1)

def text2index(complexity):
    complexity_list = ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential']
    return complexity_list.index(complexity) if complexity in complexity_list else -1


complexity_list = ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential', 'error']
error_num = 0
expertise_list = {'constant':0, 'logn':0, 'linear':0, 'nlogn':0, 'quadratic':0, 'cubic':0, 'exponential':0}

class_totals = {complexity: 0 for complexity in complexity_list}  # initialize each class

model_id = ''
model_name = sys.argv[1] # qwen, llama, deepseek, mistral
language = sys.argv[2] # java, python
option = sys.argv[3] # single, multi
data_option = sys.argv[4] # sample, full, sample_pro
generation_type = sys.argv[5] # pipeline

if sys.argv[6] == "regen":
    expertise_regeneration= sys.argv[6] # no_regen, regen
    TAG = sys.argv[8]
    global_output_path = f"../code/output_scoring/{generation_type}/{TAG}"
    
else:
    expertise_regeneration = 'no_regen'
    TAG = sys.argv[7]
    global_output_path = f"../code/output_scoring/{generation_type}/{TAG}"
    

#Find expertise
if data_option == "step2,3" and option not in not_expertise_result :
    if expertise_regeneration == "no_regen":
        expertise = sys.argv[6]
    elif expertise_regeneration == "regen":
        expertise = sys.argv[7]
        
# Measurement of No_Expertise
else:
    if model_name == "qwen":
        model_id = 'Qwen2.5-Coder-7B-Instruct-codecomplex-simple'
    elif model_name == "llama":
        model_id = 'Meta-Llama-3.1-8B-Instruct-codecomplex-simple'
    elif model_name == "deepseek":
        model_id = 'deepseek-coder-7b-instruct-v1.5-codecomplex-simple'
    elif model_name == "mistral":
        model_id = 'Mistral-7B-Instruct-v0.3-codecomplex-simple'
    elif model_name == "ministral":
        model_id = 'Ministral-8B-Instruct-2410-codecomplex-simple'
    elif model_name == "codegemma":
        model_id = 'codegemma-7b-it-codecomplex-simple'
    elif model_name == "codellama":
        model_id = 'CodeLlama-7b-Instruct-hf-codecomplex-simple'
    elif model_name == "deepseekr1-qwen":
        model_id = 'DeepSeek-R1-Distill-Qwen-7B-codecomplex-simple'
    elif model_name == "deepseekr1-llama":
        model_id = 'DeepSeek-R1-Distill-Llama-8B-codecomplex-simple'
    else:
        print(f"Invalid model name: {model_name}")


########################################################################################
#Data path
if data_option.split('-')[0] == "step1" and len(data_option.split('-')) == 2:
    response_dir_base = f"../code/output_initialize/{generation_type}/{TAG}/{option}/{data_option}/{language}/None-expertise"
elif data_option == "step2,3":
    response_dir_base = f"../code/output_initialize/{generation_type}/{TAG}/{option}/{data_option}/{language}"

### Find expertise (now not use expertise)
if option in not_expertise_result:
    if data_option.split('-')[0] == "step1" and len(data_option.split('-')) == 2:
        data_file = f'../data/codecomplex_data_sampling/{data_option}-professional/{language}-multi_test_data.jsonl'
    elif data_option == "step2,3" and  option in not_expertise_result:
        data_file = f"../data/codecomplex_data_sampling/step2,3-490/{language}-multi_test_data.jsonl"
    
    if expertise_regeneration == "no_regen":
        response_dir = f"{response_dir_base}/{model_id}"
        output_score_file_base = f"{global_output_path}/{option}/{data_option}/{model_name}"
        output_score_file = f"{output_score_file_base}/{model_name}-{language}-score_results.txt"
        output_jsonl_file = f"{output_score_file_base}/{model_name}-{language}-combined_output.jsonl"
        ensure_dir(output_score_file)
        ensure_dir(output_jsonl_file)
    else:
        response_dir = f"{response_dir_base}/{model_id}/regen"
        output_score_file_base = f"../code/output_scoring/regen/{generation_type}/{TAG}/{option}/{data_option}/{model_name}"
        output_score_file = f"{output_score_file_base}/{model_name}-{language}-score_results.txt"
        output_jsonl_file = f"{output_score_file_base}/{model_name}-{language}-combined_output.jsonl"
        ensure_dir(output_score_file)
        ensure_dir(output_jsonl_file)

### Expertise debate
elif data_option == "step2,3" and  option not in not_expertise_result:
    data_file = f"../data/codecomplex_data_sampling/step2,3-490/{language}-multi_test_data.jsonl"
    if expertise_regeneration == "no_regen":
        path_root = f"{response_dir_base}/{expertise}"
        
        for entry in os.listdir(path_root):
            if len(os.listdir(path_root)) == 1: 
                response_dir = f"{path_root}/{entry}"
            else:
                pdb.set_trace()
        output_score_file_base = f"{global_output_path}/{option}/{data_option}/{language}/{expertise}"
        output_score_file = f"{output_score_file_base}/{entry}-{language}-score_results.txt"
        output_jsonl_file = f"{output_score_file_base}/{entry}-{language}-combined_output.jsonl"
        ensure_dir(output_score_file)
        ensure_dir(output_jsonl_file)
    
    elif expertise_regeneration == "regen":
        path_root = f"{response_dir_base}/{expertise}"
        for entry in os.listdir(path_root):
            if len(os.listdir(path_root)) == 1: 
                response_dir = f"{path_root}/{entry}/regen"
            else:
                pdb.set_trace()
        output_score_file_base = f"../code/output_scoring/regen/{generation_type}/{TAG}/{option}/{data_option}/{language}/{expertise}"
        output_score_file = f"{output_score_file_base}/{entry}-{language}-score_results.txt"
        output_jsonl_file = f"{output_score_file_base}/{entry}-{language}-combined_output.jsonl"
        ensure_dir(output_score_file)
        ensure_dir(output_jsonl_file)
    
    if model_name == "":
        model_name = back_name_format(entry)

else:
    print("No search path")
    pdb.set_trace()

existing_files = sorted([f for f in os.listdir(response_dir) if f.startswith("responce_")])

if data_option.split('-')[0] == "step1" and len(data_option.split('-')) == 2:
    if data_option.split('-')[1] == "10%":
        total_data = '441'
    elif data_option.split('-')[1] == "20%":
        total_data = '882'
    elif data_option.split('-')[1] == "30%":
        total_data = '1323'
    print(f"📂 Existing files: {len(existing_files)} / {total_data}")
elif data_option == "step2,3":
    print(f"📂 Existing files: {len(existing_files)} / 490")

if not os.path.exists(data_file):
    print(f"Original file does not exist: {data_file}")
    sys.exit(1)

true_data = []
with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            true_complexity = data.get("complexity", "").lower()
            true_formatted_complexity = adjust_complexity_format(true_complexity)
            true_complexity_index = text2index(true_formatted_complexity)
            true_data.append({"true_complexity": true_complexity_index, "original_data": data})
        except json.JSONDecodeError as e:
            print(f"True data parsing failed: {e}")
########################################################################################
# Response data processing
response_data = []
json_failures = 0  # JSON parsing failure count

for i in range(len(existing_files)):
    file_path = os.path.join(response_dir, f"responce_{i:04d}.txt")
    asistent_part = ""
    json_part = ""
    
    if not os.path.exists(file_path):
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        data = ast.literal_eval(content)

        if not isinstance(data, list) or 'generated_text' not in data[0]:
            raise ValueError("Invalid data format")
    except (ValueError, SyntaxError) as e:
        print(f"File parsing failed: {file_path}, error: {e}")
        json_failures += 1
        continue
    
    try:
        #if "```json" not in generated_text:
        #    raise ValueError("JSON block not detected")
        generated_text = data[0]['generated_text']
        if model_name == "qwen":
            asistent_part = generated_text.split("<|im_start|>assistant")[1]
        elif model_name == "deepseek":
            asistent_part = generated_text.split("### Response:")[1]
        elif model_name == "llama":
            asistent_part = generated_text.split("<|start_header_id|>assistant")[1]
        elif model_name == "mistral":
             asistent_part = generated_text.split("[/INST]")[1]
        elif model_name == "ministral":
            asistent_part = generated_text.split("[/INST]")[1]
        elif model_name == "codegemma":
            asistent_part = generated_text.split("<start_of_turn>")[2]
        elif model_name == "codellama":
            asistent_part = generated_text.split("[/INST]")[1]
        elif  model_name == "deepseekr1-qwen" or model_name == "deepseekr1-llama":
            asistent_part = generated_text.split("<｜Assistant｜>")[1]
        else:
            pdb.set_trace()
        
        try:
            json_part = asistent_part.split('```json')[1].split('```')[0].strip()
        except IndexError as e:
            if '{\n    "complexity":' in asistent_part:
                start_idx = asistent_part.rindex('{\n    "complexity":')
            else:
                start_idx = asistent_part.rindex('{\n  "complexity"')
            
            try:
                end_idx = asistent_part.rindex('\n}')
                json_part = asistent_part[start_idx:end_idx+2]
            except ValueError as ve:
                if model_name == "codegemma":
                    try:
                        end_idx = asistent_part.rindex('}')
                        json_part = asistent_part[start_idx:end_idx+1]
                    except ValueError:
                        # if } is not found, use the entire content
                        json_part = asistent_part[start_idx:] + "}"
                else:
                    raise ve
        
        
        try:
            parsed_data = json.loads(json_part)
        except json.JSONDecodeError as e:
            json_part = fix_json_format(json_part, model_name)
            if json_part:
                parsed_data = json.loads(json_part)
            else:
                raise ValueError("fix_json_format failed")
    
    except Exception as e:
        print(f"\n\n{i}-Start Debugging")
        print("="*100)
        print(asistent_part)
        print(f"JSON_part parsing failed: {e}")
        print(f"⚠️ JSON parsing failed. File: {file_path}")


        # if JSON parsing failed, add default values
        complexity_index = -1
        explanation = "None"
        confidence = 0
        json_failures += 1
    
    else:
        try:
            # if llm outputs multiple json, process it
            if isinstance(parsed_data, list) and len(parsed_data) > 1:
                parsed_data = parsed_data[0]
                complexity = parsed_data.get("complexity", "").lower()
            else:
                complexity = parsed_data.get("complexity", "").lower()
        except:
            # if llm outputs multiple answers in one json, process it
            if len(parsed_data.get("complexity", "")) > 1:
                if isinstance(parsed_data.get("complexity", ""), dict):
                    complexity = parsed_data.get("complexity", "").get('time', "").lower()
                else:
                    complexity = parsed_data.get("complexity", "")[0].lower()

            
        formatted_complexity = adjust_complexity_format(complexity)
        complexity_index = text2index(formatted_complexity)
        if complexity_index == -1:
            print(f"\n\n{i}-Start Debugging")
            print("="*100)
            print(f"** complexity conversion failed ** : {complexity}")
            error_num +=1
        
        explanation = parsed_data.get("explanation", "")
        confidence = parsed_data.get("confidence", 0)
        
                
        if complexity_index == -1:
            confidence = 0
            
    true_complexity_index = true_data[i]["true_complexity"] if i < len(true_data) else -1
    exclude_keys = {"complexity", "explanation"}
    
########################################################################################
## response_dict generation
    response_dict = {}
    try:
        if "pass" in parsed_data.keys():
            response_dict["pass"] = parsed_data.get("pass", 0)
    except:
        # pdb.set_trace()
        pass
   
    
    response_dict["true_complexity"] = true_complexity_index
    response_dict["predicted_complexity"] = complexity_index
    
    try:
        if "confidence" in parsed_data.keys():
            response_dict["confidence"] = confidence
    except:
        pass
    
    response_dict["explanation"] = explanation
    response_dict["original_data"] = true_data[i]["original_data"] if i < len(true_data) else {}

    response_data.append(response_dict)
########################################################################################
## Update expertise_list
for response in response_data:
    true_complexity = response["true_complexity"]
    predicted_complexity = response["predicted_complexity"]

    if true_complexity == predicted_complexity and true_complexity != -1:
        expertise_key = complexity_list[true_complexity]
        expertise_list[expertise_key] += 1

# Calculate accuracy
val_trues = [td["true_complexity"] for td in true_data[:]]
val_preds = [rd["predicted_complexity"] for rd in response_data[:]]
valid_labels = list(range(7)) + [-1]

# Calculate the number of data for each class in true_data
for data in true_data:
    true_complexity_index = data["true_complexity"]
    if true_complexity_index != -1:  # Only calculate valid indices
        complexity_key = complexity_list[true_complexity_index]
        class_totals[complexity_key] += 1


# Model-wise accuracy
f1_per_class = f1_score(val_trues, val_preds, labels=valid_labels, average=None, zero_division=0)
expertise_json = {}
for idx, f1 in enumerate(f1_per_class):
    complexity_key = complexity_list[idx]  # Get the name of each class
    expertise_json[complexity_key] = f1  # Save the F1-score of each class
accuracy = accuracy_score(val_trues, val_preds)
f1_weighted = f1_score(val_trues, val_preds, average='macro')
f1_weighted2 = f1_score(val_trues, val_preds, average='weighted')

# Calculate the number of data for each class in true_data
class_totals = {complexity: 0 for complexity in complexity_list}
for data in true_data:
    true_complexity_index = data["true_complexity"]
    if true_complexity_index != -1:  # Only calculate valid indices
        complexity_key = complexity_list[true_complexity_index]
        class_totals[complexity_key] += 1


# Model-wise accuracy
expertise_summary = []
with open(output_score_file, 'w') as f:
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"F1 Score(macro) : {f1_weighted:.3f}\n")
    f.write(f"F1 Score(weighted) : {f1_weighted2:.3f}\n")

    f.write("\nModel-wise accuracy summary\n")
    for complexity, count in expertise_json.items():  # Based on F1-score
        total_count = class_totals.get(complexity, 0)  # Total number of data for each class
        if total_count == 0:  # Skip classes with no data
            score = "N/A"
            accuracy_ratio = 0
        else:
            accuracy_ratio = count  # Use F1-score as accuracy
            score = f"{accuracy_ratio:.4f}"

        expertise_summary.append((complexity, accuracy_ratio))  # Calculate accuracy and save
        f.write(f"{complexity.capitalize()}: {score}\n")
        
    # Calculate the highest and lowest accuracy
    expertise_summary.sort(key=lambda x: x[1], reverse=True)  # Sort by accuracy
    highest_accuracy = expertise_summary[0]
    lowest_accuracy = expertise_summary[-1]

    f.write("\nThe highest f1-score complexity:\n")
    f.write(f"{highest_accuracy[0].capitalize()} ({highest_accuracy[1] * 100:.2f}%)\n")

    f.write("\nThe lowest f1-score complexity:\n")
    f.write(f"{lowest_accuracy[0].capitalize()} ({lowest_accuracy[1] * 100:.2f}%)\n")
    
    
    labels = ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential']
    counter = Counter(val_preds)
    mapped_counts = {'constant' : 0 , 'logn': 0 , 'linear': 0 , 'nlogn': 0 , 'quadratic': 0 , 'cubic': 0 , 'exponential': 0 , -1: 0 }
    for key, count in counter.items():
        if key == -1:
            mapped_counts[-1] = count  # -1 is kept as is
        else:
            mapped_counts[labels[key]] = count  # 0~6 is converted to labels
    f.write("\noutput result count (['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential', -1])\n")
    f.write(f"{mapped_counts}\n")


label_list = val_trues
total_output = val_preds

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

# confusion matrix
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
if option == "judge_model":
    plt.title('Confusion Matrix of Debate with Judge Model', fontsize=20)
elif option == "single":
    plt.title('Confusion Matrix of Single 0-shot', fontsize=20)
elif option == "single-fewshot":
    plt.title('Confusion Matrix of Single 7-shot', fontsize=20)   
elif option == "single-no-expertise_multi-no-expertise":
    plt.title('Confusion Matrix of Debate with Majority Vote', fontsize=20) 
else:
    plt.title('Confusion Matrix of MEC³O', fontsize=20)
#plt.savefig(f"{log_dir}/{language}_MECO_confusion.png", dpi=300, bbox_inches='tight')


########################################################################################
# Output path root
if data_option.split('-')[0] == "step1" and len(data_option.split('-')) == 2:
    with open(f"{global_output_path}/{option}/{data_option}/{model_name}/{model_name}-{language}-expertise.json", "w") as t:
        json.dump(expertise_json, t, indent=4)
        plt.savefig(f"{global_output_path}/{option}/{data_option}/{model_name}/{language}_{option}_confusion.png", dpi=300, bbox_inches='tight')
        
elif data_option == "step2,3" and option not in not_expertise_result:
    with open(f"{global_output_path}/{option}/{data_option}/{language}/{expertise}/{entry}-{language}-expertise.json", "w") as t:
        json.dump(expertise_json, t, indent=4)
        plt.savefig(f"{global_output_path}/{option}/{data_option}/{language}/{expertise}/{language}_{option}_confusion.png", dpi=300, bbox_inches='tight')
        
elif data_option == "step2,3" and option in not_expertise_result:
    with open(f"{global_output_path}/{option}/{data_option}/{model_name}/{model_name}-{language}-expertise.json", "w") as t:
        json.dump(expertise_json, t, indent=4)
        plt.savefig(f"{global_output_path}/{option}/{data_option}/{model_name}/{language}_{option}_confusion.png", dpi=300, bbox_inches='tight')

# Save in JSONL format (indent=4)
with open(output_jsonl_file, 'w', encoding='utf-8') as f:
    for response in response_data:
        json.dump(response, f, ensure_ascii=False)
        f.write('\n')

# Print the number of JSON parsing failures
print("\n\n")
print("=" * 100)
print("=" * 100)

if data_option == "sample":
    print(f"⚠️ JSON parsing failed: {json_failures} / 100")
    print(f"⚠️ complexity conversion failed: {error_num} / 100")
elif data_option == "full":
    print(f"⚠️ JSON parsing failed: {json_failures} / 4900")
    print(f"⚠️ complexity conversion failed: {error_num} / 4900")
elif "1050" in data_option:
    print(f"⚠️ JSON parsing failed: {json_failures} / 1050")
    print(f"⚠️ complexity conversion failed: {error_num} / 1050")
elif data_option.split('-')[0] == "step1" and len(data_option.split('-')) == 2:
    if data_option.split('-')[1] == "10%":
        total_data = '441'
    elif data_option.split('-')[1] == "20%":
        total_data = '882'
    elif data_option.split('-')[1] == "30%":
        total_data = '1323'
    print(f"⚠️ JSON parsing failed: {json_failures} / {total_data}")
    print(f"⚠️ complexity conversion failed: {error_num} / {total_data}")
elif data_option == "step2,3":
    print(f"⚠️ JSON parsing failed: {json_failures} / 490")
    print(f"⚠️ complexity conversion failed: {error_num} / 490")    

print(f"✅ scoring completed. results are saved in {output_score_file}")
print(f"✅ response data are saved in {output_jsonl_file}")
