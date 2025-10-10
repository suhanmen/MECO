#!/bin/bash
##############################################################################################
export CUDA_VISIBLE_DEVICES=${1:-0} # gpu id

arr=("Qwen/Qwen2.5-Coder-7B-Instruct" "deepseek-ai/deepseek-coder-7b-instruct-v1.5" "mistralai/Ministral-8B-Instruct-2410")

TAG="REC"
# Default

data_option="step2,3"
# Default

option="REC-initialize"
# Default

generation_type="pipeline"
# Default

arr2=("qwen" "ministral" "deepseek")
# Default

REGEN=no_regen
# Default
##############################################################################################

# Step 1: Inference
echo "[Step 1]: Inference"
for i in "${arr[@]}"
do
    for lang in "java" "python"    
    do
        model_name=$(echo "$i" | tr '/' '_')
        log_file="../logs/${generation_type}/${data_option}/${option}/${model_name}_${lang}.log"
        mkdir -p "../logs/${generation_type}/${data_option}/${option}"
        echo "python3 ../code/sh_model.py --model $i ${@:2} --lang $lang --option $option --data_option $data_option --tag $TAG > $log_file"
        python3 ../code/sh_model.py --model "$i" "${@:2}" --lang "$lang" --option "$option" --data_option "$data_option" --tag "$TAG" > "$log_file"
    done
done


## Step2: Scoring
echo "[Step 2]: Scoring"
for i in "${arr2[@]}"
do
    for lang in "java" "python"
    do
        for option in "${option[@]}"
        do
            log_file="../code/output_scoring/${generation_type}/${TAG}/${option}/${data_option}/${i}/${i}_${lang}.log"
            mkdir -p "../code/output_scoring/${generation_type}/${TAG}/${option}/${data_option}/${i}"
            echo "python3 ../code/utils_evaluation/sh_scoring.py $i $lang $option $data_option $generation_type > $log_file"
            python3 ../code/utils_evaluation/sh_scoring.py "$i" "$lang" "$option" "$data_option" "$generation_type" "$REGEN" "$TAG" > "$log_file"
        done
    done
done

## Step3: Voting
echo "[Step 3]: Voting"
mkdir -p "../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}"
target_lang="java,python"
echo "../code/utils_evaluation/sh_voting.py ${data_option} ${option} ${generation_type} ${target_lang} ${TAG} > ../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}/voting.log"
python3 ../code/utils_evaluation/sh_voting.py \
    ${data_option} \
    ${option} \
    ${generation_type} \
    ${target_lang} \
    ${TAG} \
    > ../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}/voting.log