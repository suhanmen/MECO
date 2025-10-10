import os
import json
import numpy as np
from collections import defaultdict
import argparse
import pdb
# Script to analyze token usage per sample and per discussion-language

def collect_token_counts_from_multiple_dirs(root_dirs, category_prefix):
    """Collect token counts from multiple directories."""
    global_token_counts = [] # Total token count per sample
    # Language-wise token count per sample
    token_counts_per_lang = defaultdict(list)
    # Save round token count per discussion×language
    convo_round_totals = defaultdict(list)
    # List of dirs checked for tokens
    total_dirs = []
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            continue
        generation_file_path = os.path.join(root_dir, "generation_results.json")
        if os.path.exists(generation_file_path):
            #print(f"\t Processing model: {root_dir.split('pipeline')[1] if 'pipeline' in root_dir else root_dir}")
            total_dirs.append(root_dir)
            try:
                with open(generation_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Determine language from path
                if "java" in root_dir:
                    lang = "java"
                elif "python" in root_dir:
                    lang = "python"
                else:
                    lang = "other"

                # Sample-wise count
                for item in data.values():
                    count = item.get("generated_token_count", 0)
                    global_token_counts.append(count)
                    token_counts_per_lang[lang].append(count)

            except Exception as e:
                print(f"[Error] Failed to process {generation_file_path}: {e}")

    
    return global_token_counts, token_counts_per_lang, total_dirs

def collect_token_counts(root_dir, category_prefix):
    global_token_counts = [] # Total token count per sample
    # Language-wise token count per sample
    token_counts_per_lang = defaultdict(list)
    # Save round token count per discussion×language
    convo_round_totals = defaultdict(list)
    # List of dirs checked for tokens
    total_dirs = []
    
    # Check if root_dir contains multiple directories (for combined analysis)
    if os.path.isdir(root_dir):
        # Check if root_dir itself has generation_results.json
        generation_file_path = os.path.join(root_dir, "generation_results.json")
        if os.path.exists(generation_file_path):
            print(f"\t Processing model: {root_dir.split('pipeline')[1]}")
            total_dirs.append(root_dir)
            try:
                with open(generation_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Determine language from path
                if "java" in root_dir:
                    lang = "java"
                elif "python" in root_dir:
                    lang = "python"
                else:
                    lang = "other"

                # Sample-wise count
                for item in data.values():
                    count = item.get("generated_token_count", 0)
                    global_token_counts.append(count)
                    token_counts_per_lang[lang].append(count)


            except Exception as e:
                print(f"[Error] Failed to process {generation_file_path}: {e}")
        else:
            # Process all subdirectories in root_dir
            for model_folder in os.listdir(root_dir):
                model_folder_path = os.path.join(root_dir, model_folder)
                if not os.path.isdir(model_folder_path):
                    continue

                # Find generation_results.json file inside the model directory
                generation_file_path = os.path.join(model_folder_path, "generation_results.json")
                if not os.path.exists(generation_file_path):
                    continue
                    
                total_dirs.append(model_folder_path)
                try:
                    with open(generation_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Determine language from path
                    if "java" in root_dir:
                        lang = "java"
                    elif "python" in root_dir:
                        lang = "python"
                    else:
                        lang = "other"

                    # Sample-wise count
                    for item in data.values():
                        count = item.get("generated_token_count", 0)
                        global_token_counts.append(count)
                        token_counts_per_lang[lang].append(count)

                except Exception as e:
                    print(f"[Error] Failed to process {generation_file_path}: {e}")

    return global_token_counts, token_counts_per_lang, total_dirs


def append_stats_to_file(category, global_tc, tc_per_lang, save_path, total_dirs, root_dir):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    msg = ''
    
    # Calculate combined statistics for all languages (java + python)
    all_lang_counts = []
    for lang, counts in tc_per_lang.items():
        if counts:
            all_lang_counts.extend(counts)
    
    # Overall samples statistics (java + python combined)
    #pdb.set_trace()
    if all_lang_counts:
        msg += (
            f"[{category} All Samples]\n"
            f"  Total samples       : {len(all_lang_counts)}\n"
            f"  Total tokens        : {np.sum(all_lang_counts)}\n"
            f"  Avg. tokens/sample  : {np.mean(all_lang_counts):.2f}\n"
            f"  Std. deviation      : {np.std(all_lang_counts):.2f}\n\n"
        )
    
    # Individual language statistics
    for lang, counts in tc_per_lang.items():
        if counts:
            msg += (
                f"[{category} Samples: {lang}]\n"
                f"  Total samples       : {len(counts)}\n"
                f"  Total tokens        : {np.sum(counts)}\n"
                f"  Avg. tokens/sample  : {np.mean(counts):.2f}\n"
                f"  Std. deviation      : {np.std(counts):.2f}\n\n"
            )

    
    root_dict = {}
    for total_dir in total_dirs:
        if total_dir.split('/')[-1] not in root_dict.keys():
            root_dict[total_dir.split('/')[-1]] = {"python": [], "java": []}
        if "python" in total_dir:
            root_dict[total_dir.split('/')[-1]]["python"].append(total_dir.split('pipeline')[1].split('step2,3')[0].split('/')[2])
        elif "java" in total_dir:
            root_dict[total_dir.split('/')[-1]]["java"].append(total_dir.split('pipeline')[1].split('step2,3')[0].split('/')[2])
    
    msg += f"[{category} Root Dict]\n"
    for key, value in root_dict.items():
        for lang, dirs in value.items():
            msg += f"  {key} {lang}: {dirs}\n"
        msg += "\n"
    
    # Save all statistics to main file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",  type=str, required=False, help="Single root directory (for backward compatibility)")
    parser.add_argument("--root_dirs", type=str, nargs='+', required=False, help="Multiple root directories")
    parser.add_argument("--category",  type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    # Check which argument is provided
    if args.root_dirs:
        # Use multiple root directories
        global_tc, tc_per_lang, total_dirs = collect_token_counts_from_multiple_dirs(args.root_dirs, args.category)
        pdb.set_trace()
        append_stats_to_file(args.category, global_tc, tc_per_lang, args.save_path, total_dirs, args.root_dirs[0])
    elif args.root_dir:
        # Use single root directory (backward compatibility)
        global_tc, tc_per_lang, total_dirs = collect_token_counts(args.root_dir, args.category)
        append_stats_to_file(args.category, global_tc, tc_per_lang, args.save_path, total_dirs, args.root_dir)
    else:
        print("Error: Either --root_dir or --root_dirs must be provided")
        exit(1)
