import os
import json
import pdb

import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def find_first_pass_idx(judge_dir):
    pass_indices = {"java": {}, "python": {}}

    judge_paths = sorted([
        os.path.join(judge_dir, name)
        for name in os.listdir(judge_dir)
        if name.startswith("MAD-judge") and os.path.isdir(os.path.join(judge_dir, name))
    ])

    for judge_idx, judge_path in enumerate(judge_paths):
        for suffix in ["java", "python"]: 
            result_file = os.path.join(judge_path, f"step2,3/output-{suffix}-voting.json")
            if not os.path.isfile(result_file):
                continue

            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for sample_id, entry in data.items():
                sample_id = int(sample_id)
                pass_flag = entry.get("qwen", {}).get("pass", 0)

                if sample_id in pass_indices[suffix]:
                    continue # If the sample_id is already recorded, skip

                if pass_flag == 1:
                    pass_indices[suffix][sample_id] = {
                        'judge_idx': judge_idx + 1,
                        'True_label': entry.get("qwen", {}).get("true_complexity", 0),
                        'predict_label': entry.get("qwen", {}).get("predicted_complexity", 0)
                    }

                elif pass_flag == 0 and judge_idx == 4:
                    pass_indices[suffix][sample_id] = {
                        'judge_idx': judge_idx + 1,
                        'True_label': entry.get("qwen", {}).get("true_complexity", 0),
                        'predict_label': entry.get("qwen", {}).get("predicted_complexity", 0)
                    }


    sorted_pass_indices = {
        "java": dict(sorted(pass_indices["java"].items())),
        "python": dict(sorted(pass_indices["python"].items()))
    }

    return sorted_pass_indices


def collect_mad_token_counts(response_base_dir, pass_indices, model_name="Qwen2.5-Coder-7B-Instruct-codecomplex-simple"):    
    total_paths = []
    
    token_stats = {
        "java": defaultdict(int),
        "python": defaultdict(int)
    }

    token_sample_counts = {
        # count how many times the sample_id appears in the rounds
        "java": defaultdict(int),   
        "python": defaultdict(int)
    }

    # Round 1 is always included (for all samples)
    base_rounds = ["MAD-affirmative_discussion1", "MAD-negative_discussion1", "MAD-judge1"]

    for lang in ["java", "python"]:
        for sample_id, result in pass_indices[lang].items():
            if result is None:
                continue

            judge_idx = result["judge_idx"]

            # list of rounds to include for this sample
            relevant_rounds = base_rounds.copy()
            if judge_idx >= 2:
                for i in range(2, judge_idx + 1):
                    relevant_rounds.append(f"MAD-affirmative_discussion{i}")
                    relevant_rounds.append(f"MAD-negative_discussion{i}")
                    relevant_rounds.append(f"MAD-judge{i}")
           
            for folder in relevant_rounds:
                folder_path = os.path.join(response_base_dir, f'{folder}/step2,3/{lang}/{model_name}')
                if not os.path.isdir(folder_path):
                    continue
                total_paths.append(folder_path)
                
                for root, _, files in os.walk(folder_path):
                    if "generation_results.json" not in files:
                        continue

                    
                    detected_lang = "python" if "python" in root.lower() else "java" if "java" in root.lower() else None
                    if detected_lang != lang:
                        continue

                    file_path = os.path.join(root, "generation_results.json")
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if str(sample_id) in data:
                        token_stats[lang][sample_id] += data[str(sample_id)].get("generated_token_count", 0)
                        token_sample_counts[lang][sample_id] += 1  # ✅ Since the token is added, add 1 to the round 1
                    else:
                        only_val = next(iter(data.values()), None)
                        if isinstance(only_val, dict):
                            token_stats[lang][sample_id] += only_val.get("generated_token_count", 0)
                            token_sample_counts[lang][sample_id] += 1  # ✅ Similarly, add 1 to the round 1

    return token_stats, token_sample_counts, total_paths




def print_token_stats_from_dict(token_stats_dict, sample_count_dict, total_paths, category_name="MAD", count_by="token_units", save_path=None):
    all_counts = []
    total_sample_count = 0
    output_lines = []

    for lang in ["java", "python"]:
        token_dict = token_stats_dict.get(lang, {})
        count_dict = sample_count_dict.get(lang, {})

        counts = list(token_dict.values())
        if not counts:
            continue

        if count_by == "unique_samples":
            sample_count = len(count_dict)
        elif count_by == "token_units":
            sample_count = sum(count_dict.values())
        else:
            raise ValueError(f"Invalid count_by value: {count_by}")

        total_sample_count += sample_count
        all_counts.extend(counts)

        output_lines.append(f"[{category_name} Samples: {lang}]")
        output_lines.append(f"  Total samples       : {sample_count}")
        output_lines.append(f"  Total tokens        : {np.sum(counts)}")
        output_lines.append(f"  Avg. tokens/sample  : {np.sum(counts) / sample_count:.2f}")
        output_lines.append(f"  Std. deviation      : {np.std(counts):.2f}")
        output_lines.append("")

    if all_counts:
        output_lines.append(f"[{category_name} All Samples]")
        output_lines.append(f"  Total samples       : {total_sample_count}")
        output_lines.append(f"  Total tokens        : {np.sum(all_counts)}")
        output_lines.append(f"  Avg. tokens/sample  : {np.sum(all_counts) / total_sample_count:.2f}")
        output_lines.append(f"  Std. deviation      : {np.std(all_counts):.2f}")
        output_lines.append("")

    #print("\n".join(output_lines))
    debate_summary = {}

    for path in total_paths:
        match = re.search(r"(MAD-[^/]+)", path)
        if not match:
            continue
        round_name = match.group(1)
        
        if "affirmative" in round_name:
            stance = "affirmative"
        elif "negative" in round_name:
            stance = "negative"
        elif "judge" in round_name:
            stance = "judge"
        else:
            stance = "unknown"

        if "/python/" in path:
            lang = "python"
        elif "/java/" in path:
            lang = "java"
        else:
            continue

        if round_name not in debate_summary:
            debate_summary[round_name] = {
                "stance": stance,
                "java": 0,
                "python": 0
            }

        debate_summary[round_name][lang] += 1

    import json
    #print("\n===== Debate Round Usage Summary (by language count) =====")
    #print(json.dumps(debate_summary, indent=2))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
            f.write("\n\n===== Debate Round Usage Summary =====\n")
            f.write(json.dumps(debate_summary, indent=2)) 






def run_token_eval(pass_indices, save_dir, category_name="MAD"):
    cf_labels = [
        r'$O(1)$',         # constant
        r'$O(\log n)$',    # logn
        r'$O(n)$',         # linear
        r'$O(n \log n)$',  # nlogn
        r'$O(n^2)$',       # quadratic
        r'$O(n^3)$',       # cubic
        r'$O(2^n)$'        # exponential
    ]

    os.makedirs(save_dir, exist_ok=True)

    for lang in ["java", "python"]:
        preds = []
        trues = []

        for sid, entry in pass_indices[lang].items():
            if entry is None:
                continue
            trues.append(entry["True_label"])
            preds.append(entry["predict_label"])

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="macro")
        f1_weighted = f1_score(trues, preds, average="weighted")

        result_text = (
            f"[{category_name} Evaluation: {lang}]\n"
            f"Accuracy: {acc:.4f}\n"
            f"F1(macro): {f1:.4f}\n"
            f"F1(weighted): {f1_weighted:.4f}\n"
        )
        #print(result_text.strip())

        # confusion matrix visualization
        cm = confusion_matrix(trues, preds, labels=list(range(7)), normalize='true')
        sns.set(font_scale=1.8)
        plt.figure(figsize=(12, 9))
        ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=cf_labels, yticklabels=cf_labels)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=16)
        plt.xlabel('Predictions', fontsize=20)
        plt.ylabel('True Labels', fontsize=20)
        plt.title(f'Confusion Matrix of {category_name} ({lang})', fontsize=20)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

        plt.savefig(os.path.join(save_dir, f"{category_name}_eval_confusion_{lang}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        with open(os.path.join(save_dir, f"{category_name}_eval_result_{lang}.txt"), 'w', encoding='utf-8') as f:
            f.write(result_text)




###########################################################################
judge_dir = "../code/output_voting/pipeline/MAD"
response_dir = "../code/output_initialize/pipeline/MAD"

pass_indices = find_first_pass_idx(judge_dir)
token_stats, token_sample_counts, total_paths = collect_mad_token_counts(response_dir, pass_indices)

save_path = "../code/output_voting/pipeline/MAD/MAD_eval.txt"
print_token_stats_from_dict(token_stats, token_sample_counts, total_paths,category_name="MAD", count_by="token_units", save_path=save_path)

save_dir = "../code/output_voting/pipeline/MAD"
run_token_eval(pass_indices, save_dir)


