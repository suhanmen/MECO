import json
import os
import argparse
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

labels = ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential']

def trans_confidence(x: float) -> float:
    x = float(x)
    if x <= 0.6: return 0.1
    if 0.6 < x < 0.8: return 0.3
    if 0.8 <= x < 0.9: return 0.5
    if 0.9 <= x < 1.0: return 0.8
    if x == 1.0: return 1.0
    return x

def final_output(responses: List[Dict[str, Any]]) -> str:
    total: Dict[str, List[float]] = {label: [] for label in labels}
    for r in responses:
        idx = r.get('answer')
        conf = trans_confidence(r.get('confidence', 0.0))
        if 0 <= idx < len(labels):
            total[labels[idx]].append(conf)

    best_ans, best_score = None, -1.0
    for ans, scores in total.items():
        score_sum = sum(scores)
        if score_sum > best_score:
            best_ans = ans
            best_score = score_sum
    return best_ans or 'error'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--option", type=str, required=True)
    parser.add_argument("--data_option", type=str, required=True)
    args = parser.parse_args()

    # set path
    voting_path = f"./scripts/result/0.voting/pipeline/{args.option}/{args.data_option}/output-{args.lang}-voting.json"
    log_path = f"./scripts/result/0.voting/pipeline/{args.option}/{args.data_option}/REC_log_eval_{args.lang}.txt"

    with open(voting_path, 'r') as f:
        voting_data = json.load(f)

    y_pred, y_true = [], []

    for idx, sample in voting_data.items():
        # collect predictions from each agent
        responses = []
        for model in ["qwen", "deepseek", "ministral"]:
            if model in sample:
                pred_idx = sample[model].get("predicted_complexity", -1)
                conf = sample[model].get("confidence", 0.0)
                if pred_idx >= 0:
                    responses.append({
                        "answer": pred_idx,
                        "confidence": conf
                    })

        # true label
        true_idx = sample.get("qwen", {}).get("true_complexity", -1)  # true_complexity is saved in any agent
        if not (0 <= true_idx < len(labels)):
            continue

        pred_label = final_output(responses)
        true_label = labels[true_idx]

        y_pred.append(pred_label)
        y_true.append(true_label)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')

    # save log
    with open(log_path, "w") as f:
        f.write(f"✅ Evaluation Result for {args.lang} ({args.option}/{args.data_option})\n\n")
        f.write(f"Accuracy       : {acc:.4f}\n")
        f.write(f"Macro F1 Score : {f1:.4f}\n")


    cf_labels = [
        r'$O(1)$',         # constant
        r'$O(\log n)$',    # logn
        r'$O(n)$',         # linear
        r'$O(n \log n)$',  # nlogn
        r'$O(n^2)$',       # quadratic
        r'$O(n^3)$',       # cubic
        r'$O(2^n)$'        # exponential
    ]

    # save path
    output_path_corrected = f"./scripts/result/0.voting/pipeline/REC-discussion2/step2,3/REC_Eval_{args.lang}-confusionmatrix.png"
    os.makedirs(os.path.dirname(output_path_corrected), exist_ok=True)

    # visualization
    sns.set(font_scale=1.8)
    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=cf_labels, yticklabels=cf_labels)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=16)
    plt.xlabel('Predictions', fontsize=20)
    plt.ylabel('True Labels', fontsize=20)
    plt.title('Confusion Matrix of RECONCILE Voting', fontsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    # save
    plt.savefig(output_path_corrected, dpi=300, bbox_inches='tight')
    plt.close()

    # print result file path (for script execution check)
    print(f"✅ Confusion matrix saved to: {output_path_corrected}")





if __name__ == "__main__":
    main()

