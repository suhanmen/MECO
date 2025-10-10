import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import argparse
import pdb

def load_true_data(data_path):
    true_labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            label = item.get("complexity", "").lower()
            true_labels.append(label_to_index(label))
    return true_labels

def label_to_index(label):
    mapping = {
        "constant": 0, "logn": 1, "linear": 2, "nlogn": 3,
        "quadratic": 4, "cubic": 5, "exponential": 6, "np": 6
    }
    return mapping.get(label, -1)

def extract_pred_from_tie_response(res_path, model_name):
    # extract predicted_complexity from LLM response
    with open(res_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if model_name == "qwen":
            json_block = content.split("<|im_start|>assistant")[1]
        elif model_name == "deepseek":
            json_block = content.split("### Response:")[1]
        elif model_name == "llama":
            json_block = content.split("<|start_header_id|>assistant")[1]
        elif model_name == "ministral":
            json_block = content.split("[/INST]")[1]
        else:
            raise ValueError("Unsupported model")
        
        try:
            json_part = json_block.split('```json')[1].split('```')[0]
        except IndexError:
            start = json_block.find('{\n  "final_decision"')
            end = json_block.find('\n}', start)
            json_part = json_block[start:end+2]
        
        data = json.loads(json_part)
        pred_label = label_to_index(data.get("final_decision", "").lower())
        return pred_label

def run_eval(args):
    not_tie_path = f"../code/output_voting/pipeline/CMD/CMD-tie/{args.data_option}/CMD-Not-tie_list-{args.language}.json"
    response_dir = f"../code/output_initialize/pipeline/CMD/CMD-tie/{args.data_option}/{args.language}/{args.model_name}/"
    data_file = f"../data/codecomplex_data_sampling/{args.data_option}-490/{args.language}-multi_test_data.jsonl"

    with open(not_tie_path, 'r', encoding='utf-8') as f:
        not_tie_data = json.load(f)

    not_tie_dict = {item["idx"]: item["predicted_complexity"] for item in not_tie_data}

    true_data = load_true_data(data_file)

    predictions = []
    labels = []

    for idx in range(len(true_data)):
        true_label = true_data[idx]
        labels.append(true_label)

        if idx in not_tie_dict:
            pred = not_tie_dict[idx]
        else:
            res_path = os.path.join(response_dir, f"responce_{idx:04d}.txt")
            if os.path.exists(res_path):
                pred = extract_pred_from_tie_response(res_path, args.model_name)
            else:
                pred = -1  # failed case
        predictions.append(pred)

    
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    print(f"Accuracy: {acc:.4f} | F1(macro): {f1_macro:.4f} | F1(weighted): {f1_weighted:.4f}")
    #pdb.set_trace()
    
    # calculate confusion matrix
    cm = confusion_matrix(labels, predictions, labels=list(range(7)), normalize='true')

    cf_labels = [
        r'$O(1)$',         # constant
        r'$O(\log n)$',    # logn
        r'$O(n)$',         # linear
        r'$O(n \log n)$',  # nlogn
        r'$O(n^2)$',       # quadratic
        r'$O(n^3)$',       # cubic
        r'$O(2^n)$'        # exponential
    ]

    sns.set(font_scale=1.8)
    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=cf_labels, yticklabels=cf_labels)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=16)
    plt.xlabel('Predictions', fontsize=20)
    plt.ylabel('True Labels', fontsize=20)
    plt.title(f'Confusion Matrix of CMD Voting ({args.language})', fontsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    # save
    output_path = f"../code/output_voting/pipeline/CMD/{args.language}_CMD_eval_confusion.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_option", type=str, default="step2,3")
    args = parser.parse_args()
    run_eval(args)
