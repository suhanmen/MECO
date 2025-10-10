#!/bin/bash
##############################################################################################
export CUDA_VISIBLE_DEVICES=${1:-0} # gpu id

# You can change model name here
arr=(
    "Qwen/Qwen2.5-Coder-7B-Instruct" 
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5" 
    "mistralai/Ministral-8B-Instruct-2410"
    )

TAG="CMD" # tag for the output
# Default

option_arr=("CMD-Group1_discussion1" "CMD-Group2_discussion1" "CMD-Group1_discussion2" "CMD-Group2_discussion2") # discussion option
# Default

data_option="step2,3" # full data
#Default

generation_type="pipeline" # model generation type
#Default

REGEN=no_regen
# Default

TARGET_LANG=("java" "python") # target language
# Components: "java" "python"
##############################################################################################

for option in "${option_arr[@]}"
do
    echo "========== Processing option: $option =========="

    Step 1: Model Inference (logs)
    echo "Step 1: Model Inference (logs)"
    for model in "${arr[@]}"
    do
        for lang in "${TARGET_LANG[@]}"
        do
            model_name=$(echo "$model" | tr '/' '_')
            log_dir="../logs/${generation_type}/${data_option}/${TAG}/${option}"
            log_file="${log_dir}/${model_name}_${lang}.log"
            mkdir -p "$log_dir"

            echo "Running model: $model ($lang) with option: $option"
            python3 ../code/sh_model.py \
            --model "$model" \
            "${@:2}" \
            --lang "$lang" \
            --option "$option" \
            --data_option "$data_option" \
            --tag "$TAG" \
            > "$log_file"
        done
    done

    # Step 2: Output scoring (per model)
    echo "Step 2: Output scoring (per model)"
    for model in "${arr[@]}"
    do
        if [[ "$model" == "Qwen/"* ]]; then
            model_short="qwen"
        elif [[ "$model" == "deepseek-ai/"* ]]; then
            model_short="deepseek"
        elif [[ "$model" == "mistralai/"* ]]; then
            model_short="ministral"
        else
            echo "Unknown model format: $model"
            continue
        fi

        for lang in "${TARGET_LANG[@]}"
        do
            result_dir="../code/output_scoring/${generation_type}/${TAG}/${option}/${data_option}/${model_short}"
            log_file="${result_dir}/${model_short}_${lang}.log"
            mkdir -p "$result_dir"

            echo "Scoring model: $model_short ($lang) for option: $option"
            python3 ../code/utils_evaluation/sh_scoring.py "$model_short" "$lang" "$option" "$data_option" "$generation_type" "$REGEN" "$TAG" > "$log_file"

        done
    done

    # Step 3: Final voting for this option
    echo "Step 3: Final voting for this option"
    voting_log="../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}/voting.log"
    mkdir -p "$(dirname "$voting_log")"

    echo "Final voting for option: $option"
    python3 ../code/utils_evaluation/sh_voting.py "$data_option" "$option" "$generation_type" "$(IFS=','; echo "${TARGET_LANG[*]}")" "$TAG" > "$voting_log"

    echo "========== Finished option: $option =========="
    echo ""
done
