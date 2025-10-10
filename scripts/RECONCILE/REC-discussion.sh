#!/bin/bash
##############################################################################################
export CUDA_VISIBLE_DEVICES=${1:-0} # gpu id

TAG="REC"
# Default

arr=("Qwen/Qwen2.5-Coder-7B-Instruct" "deepseek-ai/deepseek-coder-7b-instruct-v1.5" "mistralai/Ministral-8B-Instruct-2410")
# You can change model name here

declare -A MODEL_SHORT_DICT # Dictionary for model short name
MODEL_SHORT_DICT["Qwen/Qwen2.5-Coder-7B-Instruct"]="qwen"
MODEL_SHORT_DICT["deepseek-ai/deepseek-coder-7b-instruct-v1.5"]="deepseek"
MODEL_SHORT_DICT["mistralai/Ministral-8B-Instruct-2410"]="ministral"


option_arr=("REC-discussion2") #"REC-discussion1"
# Default

data_option="step2,3"
# Default

generation_type="pipeline"
# Default

REGEN=no_regen
# Default

target_lang="java,python"
# Default
##############################################################################################

for option in "${option_arr[@]}"
do
    echo "========== Processing option: $option =========="

    ## Step 1: Model Inference (logs)
    echo "Step 1: Model Inference (logs)"
    for model in "${arr[@]}"
    do
        for lang in "java" "python"
        do
            model_name=$(echo "$model" | tr '/' '_')
            log_dir="../logs/${generation_type}/${data_option}/${TAG}/${option}"
            log_file="${log_dir}/${model_name}_${lang}.log"
            mkdir -p "$log_dir"

            echo "Running model: $model ($lang) with option: $option"
            python3 ../code/sh_model.py \
                --model "$model" \
                --lang "$lang" \
                --option "$option" \
                --data_option "$data_option" \
                --tag "$TAG" \
                --generation_type "$generation_type" \
                > "$log_file"
        done
    done

    ## Step 2: Scoring
    echo "Step 2: Scoring"
    for model in "${arr[@]}"
    do
        model_short="${MODEL_SHORT_DICT[$model]}"
        
        if [[ -z "$model_short" ]]; then
            echo "Unknown model format: $model"
            continue
        fi

        for lang in "java" "python"
        do
            result_dir="../code/output_scoring/${generation_type}/${TAG}/${option}/${data_option}/${model_short}"
            log_file="${result_dir}/${model_short}_${lang}.log"
            mkdir -p "$result_dir"

            echo "Scoring model: $model_short ($lang) for option: $option"
            python3 ../code/utils_evaluation/sh_scoring.py "$model_short" "$lang" "$option" "$data_option" "$generation_type" "$REGEN" "$TAG" > "$log_file"
        done
    done

    # Step 3: Voting
    echo "Step 3: Voting -> $option"
    voting_log="../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}/voting.log"
    mkdir -p "$(dirname "$voting_log")"
    python3 ../code/utils_evaluation/sh_voting.py "$data_option" "$option" "$generation_type" "$target_lang" "$TAG" > "$voting_log"
done


## Step 4: Evaluation
echo "Step 4: Evaluation"
for lang in "java" "python"
do
    python3 ../code/utils_evaluation/RECONCILE_eval.py \
        --lang "$lang" \
        --option "REC-discussion2" \
        --data_option "$data_option" \

    echo ""
    echo "========== Finished option: $option =========="
done


