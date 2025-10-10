import os
import json
import traceback
from sklearn.metrics import accuracy_score, f1_score
import sys
import ast
import re
import pdb

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


language_option = sys.argv[1]
data_option = sys.argv[2]
option_task = sys.argv[3]
add_expertise_classification_score = sys.argv[4]

logit_score_path = []
logit_score_root = f"./scripts/sh_response/model/{option_task}/{data_option}/{language_option}/"
logit_score_name = os.listdir(logit_score_root)
for logit_name in logit_score_name:
    logit_score_path.append(os.path.join(logit_score_root, logit_name))
    if len(os.listdir(logit_path)) == 7:
        logit_score_path = logit_path
        break


if len(os.listdir(logit_path)) == 7:
    



# --- load expertise files (load multiple files per model) ---
if language_option == "java":
    # list of expertise file paths per model
    java_expertise_paths = [
        f"./scripts/result/step1-10%/single/{i}/{i}-java-expertise.json" 
        for i in ['deepseek', 'llama', 'ministral', 'qwen']
    ]
    java_test_multi = f"./scripts/result/0.voting/{data_option}/{option_task}/output-java-voting.json"
    
    expertise_data = {}
    for path in java_expertise_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # assume the model name is the name of the upper folder of the file path (e.g. .../deepseek/deepseek-java-expertise.json → 'deepseek')
            model_name = path.split("/")[-2]
            expertise_data[model_name] = data
        except Exception as e:
            print(f"Error loading {path}: {e}")
            traceback.print_exc()
            sys.exit(1)
            
    try:
        with open(java_test_multi, "r", encoding="utf-8") as f:
            test_multi_data = json.load(f)
    except Exception as e:
        print(f"Error loading {java_test_multi}: {e}")
        traceback.print_exc()
        sys.exit(1)
    
elif language_option == "python":
    python_expertise_paths = [
        f"./scripts/result/step1-10%/single/{i}/{i}-python-expertise.json" 
        for i in ['deepseek', 'llama', 'ministral', 'qwen']
    ]
    python_test_multi = f"./scripts/result/0.voting/{data_option}/{option_task}/output-python-voting.json"
    
    expertise_data = {}
    for path in python_expertise_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # assume the model name is the name of the upper folder of the file path (e.g. .../deepseek/deepseek-java-expertise.json → 'deepseek')
            model_name = path.split("/")[-2]
            expertise_data[model_name] = data
        except Exception as e:
            print(f"Error loading {path}: {e}")
            traceback.print_exc()
            sys.exit(1)
            
    try:
        with open(python_test_multi, "r", encoding="utf-8") as f:
            test_multi_data = json.load(f)
    except Exception as e:
        print(f"Error loading {python_test_multi}: {e}")
        traceback.print_exc()
        sys.exit(1)

# --- calculate voting and final prediction ---
total_output = {}

# assume the key of test_multi_data is "0", "1", ... (string)
for idx in range(len(test_multi_data)):
    votes = []             # list to store vote (prediction and expertise_score) per model
    expertise_scores = []  # total expertise_score (for debugging or separate use)
    
    case_data = test_multi_data[str(idx)]
    label_keys = list(case_data.keys())
    
    # assume the actual answer (label) is included in the second key
    try:
        label = case_data[label_keys[1]]['true_complexity']
    except Exception as e:
        print(f"Error retrieving label for case {idx}: {e}")
        pdb.set_trace()
        label = None
    
    total_output[str(idx)] = {}  # initialize dictionary to store result of the test case
    
    for model_output in case_data.keys():
        # skip 'vote' key
        if model_output != 'vote':
            parts = model_output.split('-')
            if len(parts) < 2:
                print(f"Unexpected model_output format: {model_output}")
                pdb.set_trace()
                continue
           
            # first part: expertise label (expected time complexity label)
            expertise_part = parts[0]
            # second part: model name
            model_name = parts[1]
            # prediction result of the model (e.g. 'linear', 'nlogn' etc.)
            
            predicted_complexity = case_data[model_output]["predicted_complexity"]
            
            try:
                # expertise_data[model_name] is the dictionary structure loaded above
                # expertise JSON is in the form of { "constant": float, "linear": float, ... }
                # get expertise_score from expertise_data[model_name][expertise_part]
                expertise_score = expertise_data[model_name][label_format(predicted_complexity)]
            except Exception as e:
                #print(f"Error processing expertise for {model_output}: {e}")
                expertise_score = 0  # if error occurs, process as 0
             
            votes.append({
                "model": model_name,
                "expertise": expertise_part,
                "predicted_complexity": predicted_complexity,
                "expertise_score": expertise_score
            })
            expertise_scores.append(expertise_score)
    
    # calculate the sum of expertise_score per predicted_complexity per test case
    aggregated_scores = {}
    for vote in votes:
        pred = vote["predicted_complexity"]
        score = vote["expertise_score"] if vote["expertise_score"] != 'error' else 0
        aggregated_scores[pred] = aggregated_scores.get(pred, 0) + score
    
    
    # select the predicted_complexity with the highest expertise_score as the final prediction
    if aggregated_scores:
        final_prediction = max(aggregated_scores, key=aggregated_scores.get)
    else:
        final_prediction = None
        pdb.set_trace()
    
    total_output[str(idx)]["votes"] = votes
    total_output[str(idx)]["expertise_scores"] = expertise_scores
    total_output[str(idx)]["label"] = label
    total_output[str(idx)]["final_prediction"] = final_prediction

# construct true and final prediction lists for all test cases
true_labels = []
pred_labels = []
for idx in total_output:
    true_labels.append(total_output[idx]["label"])
    pred_labels.append(total_output[idx]["final_prediction"])

# accuracy and f1 score calculation (f1 is calculated by average="macro" for multi-class)
accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average="macro")

print("Accuracy:", accuracy)
print("F1 Score:", f1)


# --- save result to file --- 
output_dir = f"../output_voting/{option_task}/{data_option}/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{language_option}-total_score.json")
try:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(total_output, f, indent=4)
    print(f"Results saved to {output_file}")
except Exception as e:
    print(f"Error saving file {output_file}: {e}")
    traceback.print_exc()



