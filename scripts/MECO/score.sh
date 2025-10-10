#!/bin/bash
##############################################################################################
TAG="${1:-MECO}" # what you want to name the output directory
# Components: "ONE_ROLE_PER_SETTING" "MECO" "RECONCILE" "CMD" "MAD"

data_option="${2:-step2,3}" 

option_task="${3:-multi-codecomplex-expertise}"  
# 1. single-codecomplex-expertise -> step2,3 -> multi-codecomplex-expertise
# single-codecomplex-expertise, multi-codecomplex-expertise, single-no-expertise_multi-expertise, multi-codecomplex-expertise-with-confidence
 
generation_type="pipeline" 
# Default

expertise_regeneration='no_regen' 
# no_regen, regen

target_lang=("java" "python")
# Default

arr=("qwen" "mistral" "ministral" "deepseek" "llama" "codegemma" "codellama") # expertise order
# "qwen" "mistral" "ministral" "deepseek" "llama" "codegemma" "codellama"

not_use_voting=("judge_model-Ns_Nm" "judge_model-Ys_Nm" "judge_model-Ns_Ym" "judge_model-Ys_Ym" "single-fewshot" "single") # This setting is not used for voting.py
##############################################################################################

declare -A model_expertise_map_java
declare -A model_expertise_map_python

if [ "${TAG}" == "ONE_ROLE_PER_SETTING" ]; then
    # java - ONE_ROLE_PER_SETTING
    model_expertise_map_java["qwen"]="constant"
    model_expertise_map_java["mistral"]="logn"
    model_expertise_map_java["ministral"]="linear"
    model_expertise_map_java["deepseek"]="nlogn"
    model_expertise_map_java["llama"]="quadratic"
    model_expertise_map_java["codegemma"]="cubic"
    model_expertise_map_java["codellama"]="exponential"

    # python - ONE_ROLE_PER_SETTING
    model_expertise_map_python["qwen"]="constant"
    model_expertise_map_python["llama"]="logn"
    model_expertise_map_python["ministral"]="linear"
    model_expertise_map_python["deepseek"]="nlogn"
    model_expertise_map_python["mistral"]="quadratic"
    model_expertise_map_python["codegemma"]="cubic"
    model_expertise_map_python["codellama"]="exponential"
fi


# step1 : make score file
for i in "${arr[@]}"
do
    for lang in  "${target_lang[@]}" ;
    do
        if [ "${TAG}" == "ONE_ROLE_PER_SETTING" ]; then
            if [ "$lang" == "java" ]; then
                expertise_array=("${model_expertise_map_java[$i]}")
            elif [ "$lang" == "python" ]; then
                expertise_array=("${model_expertise_map_python[$i]}")
            fi
        elif [ "${TAG}" == "MECO" ] && [ "${option_task}" != "judge_model-Ns_Nm" ] && [ "${option_task}" != "judge_model-Ys_Nm" ] && [ "${option_task}" != "judge_model-Ns_Ym" ] && [ "${option_task}" != "judge_model-Ys_Ym" ]; then
            expertise_json_file="../code/output_scoring/pipeline/${TAG}/single/step1-10%/${lang}-expertise.json"
            if [ -f "$expertise_json_file" ]; then
                mapfile -t expertise_array < <(
                jq -r --arg m "$i" '.[$m]["time-complex"] // [] | .[] | select(type=="string" and length>0)' \
                    "$expertise_json_file" 2>/dev/null
                )
            fi
        else
            expertise_array=("None-expertise")
        fi
        
        for expertise_part in "${expertise_array[@]}"; do
            echo "i: $i"
            echo "lang: $lang"
            echo "expertise_part: $expertise_part"
            
            if [ "${expertise_regeneration}" == "regen" ]; then
                log_file="../code/output_scoring/${expertise_regeneration}/${generation_type}/${TAG}/${option_task}/${data_option}/${lang}/${expertise_part}/${lang}.log"
                mkdir -p "../code/output_scoring/${expertise_regeneration}/${generation_type}/${TAG}/${option_task}/${data_option}/${lang}/${expertise_part}"
                echo "python3 ../code/utils_evaluation/sh_scoring.py $i $lang $option_task $data_option $generation_type ${expertise_regeneration} $expertise_part $TAG > $log_file"
                python3 ../code/utils_evaluation/sh_scoring.py "$i" "$lang" "$option_task" "$data_option" "$generation_type" "$expertise_regeneration" "$expertise_part" "$TAG" > "$log_file"
            
            
            else
                log_file="../code/output_scoring/${generation_type}/${TAG}/${option_task}/${data_option}/${lang}/${expertise_part}/${lang}.log"
                mkdir -p "../code/output_scoring/${generation_type}/${TAG}/${option_task}/${data_option}/${lang}/${expertise_part}"
                echo "python3 ../code/utils_evaluation/sh_scoring.py $i $lang $option_task $data_option $generation_type $expertise_part $TAG > $log_file"
                python3 ../code/utils_evaluation/sh_scoring.py "$i" "$lang" "$option_task" "$data_option" "$generation_type" "$expertise_part" "$TAG" > "$log_file"

            fi
        done
    done
done


if [ "${option_task}" == "single" ]; then
    joined_arr=$(IFS=','; echo "${arr[*]}")
    echo "../code/utils_evaluation/sh_concat-expertise.py  ${data_option} ${generation_type} ${option_task} \"${joined_arr}\" ${TAG} > ../code/output_scoring/${generation_type}/${TAG}/${option_task}/${data_option}/concat-expertise.log"
    python3 ../code/utils_evaluation/sh_concat-expertise.py  ${data_option} ${generation_type} ${option_task} ${joined_arr} ${TAG} > ../code/output_scoring/${generation_type}/${TAG}/${option_task}/${data_option}/concat-expertise.log  

elif [ "${option_task}" in "${not_use_voting[@]}" ]; then
    echo "This setting is not used for voting."
    
else :
    ##step2 : make voting file
    if [ "${expertise_regeneration}" == "regen" ]; then
        mkdir -p ../code/output_voting/${generation_type}/${TAG}/${option_task}/${expertise_regeneration}/${data_option}/
        echo "../code/utils_evaluation/sh_voting.py ${data_option} ${option_task} ${generation_type} ${TAG} ${expertise_regeneration}  > ../code/output_voting/${expertise_regeneration}/${generation_type}/${TAG}/${option_task}/${data_option}/voting.log"
        python3 ../code/utils_evaluation/sh_voting.py  ${data_option} ${option_task} ${generation_type} "$(IFS=','; echo "${target_lang[*]}")" ${TAG} ${expertise_regeneration}   > ../code/output_voting/${expertise_regeneration}/${generation_type}/${TAG}/${option_task}/${data_option}/voting.log

    else
        mkdir -p ../code/output_voting/${generation_type}/${TAG}/${option_task}/${data_option}/
        echo "../code/utils_evaluation/sh_voting.py ${data_option} ${option_task} ${generation_type} ${TAG} > ../code/output_voting/${generation_type}/${TAG}/${option_task}/${data_option}voting.log"
        python3 ../code/utils_evaluation/sh_voting.py  ${data_option} ${option_task} ${generation_type} "$(IFS=','; echo "${target_lang[*]}")" ${TAG} > ../code/output_voting/${generation_type}/${TAG}/${option_task}/${data_option}/voting.log
    fi

    # #step3 : last for consensus weight score
    # add_weight=False
    # for lang in "java" "python"
    # do
    #     echo ".../code/utils_evaluation/sh_weight_consensus.py ${lang} ${data_option} ${option_task} ${add_weight} ${expertise_regeneration} > ../code/output_scoring/pipeline/${option_task}/${data_option}/"
    #     python ../code/utils_evaluation/sh_weight_consensus.py ${lang} ${data_option} ${option_task} ${add_weight} ${expertise_regeneration} 
    # done  


    #step4 : last for consensus weight score
    if [ "${option_task}" == "multi-codecomplex-expertise" ] && [ "${generation_type}" == "pipeline" ]; then
        for lang in "${target_lang[@]}"
        do
            echo ".../code/utils_evaluation/sh_weight_consensus_plus.py ${lang} ${data_option} ${option_task} ${expertise_regeneration} > ../code/output_scoring/pipeline/${option_task}/${data_option}/"
            python ../code/utils_evaluation/sh_weight_consensus_plus.py ${lang} ${data_option} ${option_task} ${expertise_regeneration} 
        done  
    fi
fi