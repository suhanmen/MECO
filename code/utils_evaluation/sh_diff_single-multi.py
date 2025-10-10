import json
import os
import pdb
import sys

# multi and single folder path
option_task = sys.argv[1]

'''
single no || single_no-multi_no
- ./code/output_voting/step2,3/pipeline/single
- ./code/output_voting/step2,3/pipeline/single-no-expertise_multi-no-expertise

single no || single_no-multi_yes
- ./code/output_voting/step2,3/pipeline/single
- ./code/output_voting/step2,3/pipeline/single-no-expertise_multi-expertise


single yes || single_yes-multi_no
- ./code/output_voting/step2,3/pipeline/single-codecomplex-expertise
- ./code/output_voting/step2,3/pipeline/single-expertise_multi-no-expertise

single yse || single_yes-multi_yes
- ./code/output_voting/step2,3/pipeline/single-codecomplex-expertise
- ./code/output_voting/step2,3/pipeline/mutli-codecomplex-expertise
'''
    
if option_task == 'n,n':
    format_path = ['./code/output_voting/step2,3/pipeline/single', './code/output_voting/step2,3/pipeline/single-no-expertise_multi-no-expertise']
elif option_task == 'n,y':
    format_path = ['./code/output_voting/step2,3/pipeline/single', './code/output_voting/step2,3/pipeline/single-no-expertise_multi-expertise']
elif option_task == 'y,n':
    format_path = ['./code/output_voting/step2,3/pipeline/single-codecomplex-expertise', './code/output_voting/step2,3/pipeline/single-expertise_multi-no-expertise']
elif option_task == 'y,y':   
    format_path = ['./code/output_voting/step2,3/pipeline/single-codecomplex-expertise', './code/output_voting/step2,3/pipeline/multi-codecomplex-expertise']
else:
    print('Invalid option_task')
    sys.exit(1)

def load_voting_data():
    voting_data = {"multi": {}, "single": {}}
    
    for path in format_path:
        path_key = "multi" if "multi" in path else "single"
        for lang in ["java", "python"]:
            voting_file = os.path.join(path, f"output-{lang}-voting.json")
            if os.path.exists(voting_file):
                with open(voting_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                voting_data[path_key][lang] = data
            else:
                print(f"X {voting_file} file does not exist.")
                pdb.set_trace()
    return voting_data

def compare_single_multi(voting_data):
    comparisons = {}
    # dictionary to aggregate statistics per expertise-model key (per language)
    diff_counts = {lang: {} for lang in ["java", "python"]}
    same_counts = {lang: {} for lang in ["java", "python"]}
    multi_correct_counts = {lang: {} for lang in ["java", "python"]}
    multi_wrong_counts = {lang: {} for lang in ["java", "python"]}
    
    for lang in ["java", "python"]:
        if lang in voting_data["multi"] and lang in voting_data["single"]:
            multi_data = voting_data["multi"][lang]
            single_data = voting_data["single"][lang]
            comparisons[lang] = {}
            
            # assume the example key is "0", "1", ... (string)
            for example_idx in multi_data.keys():
                example_diff = {}
                example_same = {}
                example_multi_correct = {}
                example_multi_wrong = {}
                
                # iterate through all keys in multi_data (except "vote")
                for key in multi_data[example_idx]:
                    if key == "vote":
                        continue
                    pdb.set_trace() 
                    # single data must have the same key to compare
                    if key not in single_data[example_idx]:
                        continue
                    
                    
                    multi_pred = multi_data[example_idx][key]["predicted_complexity"]
                    single_pred = single_data[example_idx][key]["predicted_complexity"]
                    true_complexity = multi_data[example_idx][key]["true_complexity"]
                    
                    if multi_pred != single_pred:
                        example_diff[key] = {"single_pred": single_pred, "multi_pred": multi_pred}
                        diff_counts[lang][key] = diff_counts[lang].get(key, 0) + 1
                        if multi_pred == true_complexity:
                            example_multi_correct[key] = multi_pred
                            multi_correct_counts[lang][key] = multi_correct_counts[lang].get(key, 0) + 1
                        elif single_pred == true_complexity:
                            example_multi_wrong[key] = {"single_pred": single_pred, "multi_pred": multi_pred}
                            multi_wrong_counts[lang][key] = multi_wrong_counts[lang].get(key, 0) + 1
                    else:
                        example_same[key] = multi_pred
                        same_counts[lang][key] = same_counts[lang].get(key, 0) + 1
                        
                if example_diff:
                    comparisons[lang][int(example_idx)] = {
                        "different_predictions": example_diff,
                        "same_predictions": example_same,
                        "multi_correct_predictions": example_multi_correct,
                        "multi_wrong_predictions": example_multi_wrong
                    }
    return comparisons, diff_counts, same_counts, multi_correct_counts, multi_wrong_counts

# Load data
voting_data = load_voting_data()

# Compare single and multi per expertise-model key
comparisons, diff_counts, same_counts, multi_correct_counts, multi_wrong_counts = compare_single_multi(voting_data)

# Print comparison results
for lang, examples in comparisons.items():
    print(f"{lang} Comparison Results (single vs multi) per expertise-model:")
    total_diff = sum(len(item["different_predictions"]) for item in examples.values())
    total_same = sum(len(item["same_predictions"]) for item in examples.values())
    total_correct = sum(len(item.get("multi_correct_predictions", {})) for item in examples.values())
    total_wrong = sum(len(item.get("multi_wrong_predictions", {})) for item in examples.values())
    
    print(f"Total Different Predictions: {total_diff}")
    print(f"Total Same Predictions: {total_same}")
    print(f"Total Multi Predictions Matching True Complexity: {total_correct}")
    print(f"Total Single Correct → Multi Wrong: {total_wrong}\n")
    
    print("Expertise-Model-wise Statistics:")
    for key in diff_counts[lang].keys() | same_counts[lang].keys() | multi_correct_counts[lang].keys() | multi_wrong_counts[lang].keys():
        d = diff_counts[lang].get(key, 0)
        s = same_counts[lang].get(key, 0)
        c = multi_correct_counts[lang].get(key, 0)
        w = multi_wrong_counts[lang].get(key, 0)
        print(f"{key}: Different={d}, Same={s}, Multi_Correct={c}, Single_Correct→Multi_Wrong={w}")
    print("="*100)
