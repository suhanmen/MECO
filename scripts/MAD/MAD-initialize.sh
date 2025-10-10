#!/bin/bash
##############################################################################################
export CUDA_VISIBLE_DEVICES=${1:-0} # gpu id

TAG="MAD"
# Default

arr="Qwen/Qwen2.5-Coder-7B-Instruct" # You can change model name here
# Default

if [ "$arr" = "Qwen/Qwen2.5-Coder-7B-Instruct" ]; then
    arr2="qwen"
elif [ "$arr" = "?" ]; then
    arr2="?"
else
    echo "Invalid model name"
    exit 1
fi


data_option="step2,3" # You can change data option here
# Default

generation_type="pipeline" # You can change generation type here
# Default

options=(
    "MAD-affirmative_discussion1" "MAD-negative_discussion1" "MAD-judge1"
    "MAD-affirmative_discussion2" "MAD-negative_discussion2" "MAD-judge2" 
    "MAD-affirmative_discussion3" "MAD-negative_discussion3" "MAD-judge3"
    "MAD-affirmative_discussion4" "MAD-negative_discussion4" "MAD-judge4"
    "MAD-affirmative_discussion5" "MAD-negative_discussion5" "MAD-judge5"
    ) 

REGEN=no_regen
# Default

REUSE=False # True (resume from the last run) or False (start from the beginning)
#Default

TARGET_LANG=("java" "python")
#############################################################################################
for option in "${options[@]}"
do
    echo "======== Running option: $option ========"

    ## STEP 1: Generation (sh_model.py)
    for lang in "java" "python"
    do
        log_dir="../logs/${generation_type}/${data_option}/${TAG}/${option}"
        log_file="${log_dir}/$(basename "${arr}")_${lang}.log"
        mkdir -p "$log_dir"
        echo "[STEP 1] python3 ../code/sh_model.py --model $arr --lang $lang --option $option --data_option $data_option"
        python3 ../code/sh_model.py --model "$arr" "${@:2}" --lang "$lang" --option "$option" --data_option "$data_option" --tag "$TAG" --reuse "$REUSE" > "$log_file"
    done

    ## STEP 2: Scoring (sh_scoring.py)
    for lang in "java" "python"
    do
        log_file="../code/output_scoring/${generation_type}/${TAG}/${option}/${data_option}/${lang}.log"
        mkdir -p "../code/output_scoring/${generation_type}/${TAG}/${option}/${data_option}/${lang}"
        echo "[STEP 2] python3 ../code/utils_evaluation/sh_scoring.py $arr2 $lang $option $data_option $generation_type $REGEN $TAG"
        python3 ../code/utils_evaluation/sh_scoring.py "$arr2" "$lang" "$option" "$data_option" "$generation_type" "$REGEN" "$TAG" > "$log_file"
    done

    ## STEP 3: Voting (sh_voting.py)
    mkdir -p "../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}/"
    echo "[STEP 3] python3 ../code/utils_evaluation/sh_voting.py $data_option $option $generation_type $(IFS=','; echo "${TARGET_LANG[*]}") $TAG"
    python3 ../code/utils_evaluation/sh_voting.py "$data_option" "$option" "$generation_type" "$(IFS=','; echo "${TARGET_LANG[*]}")" "$TAG" > "../code/output_voting/${generation_type}/${TAG}/${option}/${data_option}/voting.log"
done

## STEP 4: evaluation
echo "[STEP 4] python3 ../code/utils_evaluation/MAD_eval.py"
python3 ../code/utils_evaluation/MAD_eval.py