#!/bin/bash
##############################################################################################
export CUDA_VISIBLE_DEVICES=${1:-0} # gpu id

arr=("Qwen/Qwen2.5-Coder-7B-Instruct") # model list
# Default

option_arr=("CMD-tie") # tie option
# Default

data_option="step2,3" # full data
# Default

generation_type="pipeline" # model generation type
# Default

TAG="CMD"
# Default

TARGET_LANG=("java" "python") # target language
# Components: "java" "python"

##############################################################################################
# for option in "${option_arr[@]}"
# do
#     echo "========== Processing option: $option =========="

#     # Step 1: Model Inference (logs)
#     for model in "${arr[@]}"
#     do
#         for lang in "${TARGET_LANG[@]}"
#         do
#             model_name=$(echo "$model" | tr '/' '_')
#             log_dir="../logs/${generation_type}/${data_option}/${TAG}/${option}"
#             log_file="${log_dir}/${model_name}_${lang}.log"
#             mkdir -p "$log_dir"

#             echo "Running model: $model ($lang) with option: $option"
#             python3 ../code/sh_model.py --model "$model" "${@:2}" --lang "$lang" --option "$option" --data_option "$data_option" --tag "$TAG" > "$log_file"
#         done
#     done
# done

# Step 2: Output CMD EVAL (per model), with log
echo "Step 2: Output CMD EVAL (per model), with log"
for lang in "java" "python"
do
    echo "Running CMD_eval for $lang"
    python ../code/utils_evaluation/CMD_eval.py \
        --language $lang \
        --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
        --data_option step2,3 \
        > ../code/output_voting/pipeline/CMD/eval_log_${lang}.txt

    echo "Saved log to eval_log_${lang}.txt"
done
