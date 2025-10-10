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

data_option="step2,3" # full data
# Default

generation_type="pipeline" # model generation type
# Default

options=("CMD-Group1" "CMD-Group2") # discussion option
# Default

arr2=("qwen" "ministral" "deepseek") # model short name
# Default

REGEN=no_regen
# Default

SCORING="True"

TARGET_LANG=("java" "python") # target language
# Components: "java" "python"
##############################################################################################
## Step 1: Generation
echo "Step 1: Generation"
for option in "${options[@]}"
do
    for i in "${arr[@]}"
    do
        for lang in "${TARGET_LANG[@]}"
        do
            model_name=$(echo "$i" | tr '/' '_')
            log_dir="../logs/${generation_type}/${data_option}/${TAG}/${option}"
            log_file="${log_dir}/${model_name}_${lang}.log"
            mkdir -p "$log_dir"
            echo "python3 ../code/sh_model.py --model $i ${@:2} --lang $lang --option $option --data_option $data_option --tag $TAG > $log_file"
            python3 ../code/sh_model.py --model "$i" "${@:2}" --lang "$lang" --option "$option" --data_option "$data_option" --tag "$TAG" > "$log_file"
        done
    done
done

## Step 2: Scoring
echo "Step 2: Scoring"
if [ "${SCORING}" == "True" ]; then
    #step2: scoring
    for i in "${arr2[@]}"
    do
        for lang in "${TARGET_LANG[@]}"
        do
            for option in "${options[@]}"
            do
                log_file="../code/output_scoring/${generation_type}/${TAG}/${option}/${data_option}/${i}/${i}_${lang}.log"
                mkdir -p ../code/output_scoring/${generation_type}/${TAG}/${option}/${data_option}/${i}
                echo "python3 ../code/utils_evaluation/sh_scoring.py $i $lang $option $data_option $generation_type $REGEN $TAG > $log_file"
                python3 ../code/utils_evaluation/sh_scoring.py "$i" "$lang" "$option" "$data_option" "$generation_type" "$REGEN" "$TAG" > "$log_file"
            done
        done
    done


## Step 3: Voting
echo "Step 3: Voting"
    for option in "${options[@]}"
    do
        mkdir -p ../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}/
        echo "../code/utils_evaluation/sh_voting.py ${data_option} ${option} ${generation_type} $(IFS=','; echo "${TARGET_LANG[*]}") ${TAG} > ../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}/voting.log"
        python3 ../code/utils_evaluation/sh_voting.py  ${data_option} ${option} ${generation_type} "$(IFS=','; echo "${TARGET_LANG[*]}")" "$TAG" > ../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}/voting.log
    done
fi