#!/bin/bash
##############################################################################################
export CUDA_VISIBLE_DEVICES=${1:-0} # gpu id

TAG="${2:-MECO}" # what you want to name the output directory
# Components: "ONE_ROLE_PER_SETTING" "MECO" "RECONCILE" "CMD" "MAD"

DATA_OPTION="${3:-step2,3}" # what you want to use data option
# Components: "step1-10%" "step1-20%" "step1-30%" "step2,3"

OPTION="${4:-single-codecomplex-expertise}" # what you want to use option for generation
# Components: "single", "single-codecomplex-expertise", "multi-codecomplex-expertise"  ...
# You can find all options in code/utils_Instruction/MECO.py

GENERATION_TYPE="${5:-pipeline}" # "pipeline" (do not make logit score), "model" (make logit score)   
# Components: "pipeline" "model"

if [ $# -ge 6 ]; then
    MODELS=("${@:6}")
else
    #MODELS=("Qwen/Qwen2.5-Coder-7B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "mistralai/Ministral-8B-Instruct-2410" "deepseek-ai/deepseek-coder-7b-instruct-v1.5" "meta-llama/Meta-Llama-3.1-8B-Instruct" "google/codegemma-7b-it" "codellama/CodeLlama-7b-Instruct-hf" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    MODELS=("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
fi
# Components: "Qwen/Qwen2.5-Coder-7B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "mistralai/Ministral-8B-Instruct-2410" "deepseek-ai/deepseek-coder-7b-instruct-v1.5" "meta-llama/Meta-Llama-3.1-8B-Instruct" "google/codegemma-7b-it" "codellama/CodeLlama-7b-Instruct-hf"
# if you want to find expertise, use "step1-10%" "step1-20%" "step1-30%". if you want to evaluate, use "step2,3"

EXPER_REGENERATION='no_regen' # if model regenerate output, use "regen" and "no_regen" is not used
# Components: "no_regen" "regen"

JSON_FILE_PATH="../code/output_scoring/pipeline/${TAG}/single/step1-10%" # Load Expertise role from this json file
# Default

LOG_FILE_PATH="../logs/${GENERATION_TYPE}/${DATA_OPTION}/${TAG}/${OPTION}/${EXPER_REGENERATION}"
# Default

ONE_ROLE_PER_SETTING="False" # True (one role per setting) or False (multiple roles per setting)
# Components: "True" "False"

ONE_ROLE_DATA="java" # java or python : one role per data or multiple roles per data
# Components: "java" "python"

REUSE="False" # True (resume from the last run) or False (start from the beginning)

ALL_DATA=("java" "python") # all data
# Components: "java" "python"
##############################################################################################

### This is for multiple roles per setting
if [ "${OPTION}" == "judge_model-Ns_Nm" ] || [ "${OPTION}" == "judge_model-Ys_Nm" ] || [ "${OPTION}" == "judge_model-Ns_Ym" ] || [ "${OPTION}" == "judge_model-Ys_Ym" ]; then
    for lang in  "${ALL_DATA[@]}"; do 
        for i in "${MODELS[@]}"; do
            model_name=$(echo "$i" | tr '/' '_')
            log_file="${LOG_FILE_PATH}/${model_name}_${lang}.log"
            mkdir -p "${LOG_FILE_PATH}"
            echo "Running: python3 ../code/sh_model.py --model $i ${@:2} --lang $lang --option $OPTION --data_option $DATA_OPTION --expertise_regeneration $EXPER_REGENERATION --generation_type $GENERATION_TYPE --tag $TAG --reuse $REUSE -> $log_file"
            python3 ../code/sh_model.py \
            --model "$i" \
            --lang "$lang" \
            --option "$OPTION" \
            --data_option "$DATA_OPTION" \
            --expertise "None-expertise" \
            --expertise_regeneration "$EXPER_REGENERATION" \
            --generation_type "$GENERATION_TYPE" \
            --tag "$TAG" \
            --reuse "$REUSE" > "$log_file"
        done
    done

elif [ "${ONE_ROLE_PER_SETTING}" == "False" ]; then
    declare -A model_map    
    model_map["Qwen/Qwen2.5-Coder-7B-Instruct"]="qwen"
    model_map["mistralai/Mistral-7B-Instruct-v0.3"]="mistral"
    model_map["mistralai/Ministral-8B-Instruct-2410"]="ministral"
    model_map["deepseek-ai/deepseek-coder-7b-instruct-v1.5"]="deepseek"
    model_map["meta-llama/Meta-Llama-3.1-8B-Instruct"]="llama"
    model_map["google/codegemma-7b-it"]="codegemma"
    model_map["codellama/CodeLlama-7b-Instruct-hf"]="codellama"
    model_map["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]="deepseekr1-qwen"
    model_map["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseekr1-llama"
    
    for lang in  "${ALL_DATA[@]}"; do 
        for i in "${MODELS[@]}"; do
            json_path="${JSON_FILE_PATH}/${lang}-expertise.json" # Load Expertise role from this json file
            model_key=${model_map[$i]} # Map model name to short key

            if [ "${OPTION}" == "single" ]; then
                expertise_part=("None-expertise")
            else
                expertise_part=($(jq -r --arg model "$model_key" '.[$model]["time-complex"][]' "$json_path"))
                if [ ${#expertise_part[@]} -eq 0 ]; then # If the model is not in the json file, skip
                    echo "Info: No expertise found for $i in $json_path. Skipping $lang."
                    continue
                fi
            fi

            for expertise in "${expertise_part[@]}"; do
                model_name=$(echo "$i" | tr '/' '_')
                log_file="${LOG_FILE_PATH}/${model_name}_${lang}_${expertise}.log"
                mkdir -p "${LOG_FILE_PATH}"
                echo "Running: python3 ../code/sh_model.py --model $i ${@:2} --lang $lang --option $OPTION --data_option $DATA_OPTION --expertise $expertise --expertise_regeneration $EXPER_REGENERATION --generation_type $GENERATION_TYPE --tag $TAG --reuse $REUSE -> $log_file"
                python3 ../code/sh_model.py \
                --model "$i" \
                --lang "$lang" \
                --option "$OPTION" \
                --data_option "$DATA_OPTION" \
                --expertise "$expertise" \
                --expertise_regeneration "$EXPER_REGENERATION" \
                --generation_type "$GENERATION_TYPE" \
                --tag "$TAG" \
                --reuse "$REUSE" > "$log_file"
            done
        done
    done


### This is for one role per setting
else
    for i in "${MODELS[@]}"; do
        if [ "${ONE_ROLE_PER_SETTING}" == "True" ] && [ "${ONE_ROLE_DATA}" == "java" ]; then
            if [ "${i}" == "Qwen/Qwen2.5-Coder-7B-Instruct" ]; then
                expertise="constant"
            elif [ "${i}" == "mistralai/Mistral-7B-Instruct-v0.3" ]; then
                expertise="logn"
            elif [ "${i}" == "mistralai/Ministral-8B-Instruct-2410" ]; then
                expertise="linear"
            elif [ "${i}" == "deepseek-ai/deepseek-coder-7b-instruct-v1.5" ]; then
                expertise="nlogn"
            elif [ "${i}" == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]; then
                expertise="quadratic"
            elif [ "${i}" == "google/codegemma-7b-it" ]; then
                expertise="cubic"
            elif [ "${i}" == "codellama/CodeLlama-7b-Instruct-hf" ]; then
                expertise="exponential"
            fi
        
        elif [ "${ONE_ROLE_PER_SETTING}" == "True" ] && [ "${ONE_ROLE_DATA}" == "python" ]; then
            if [ "${i}" == "Qwen/Qwen2.5-Coder-7B-Instruct" ]; then
                expertise="constant"
            elif [ "${i}" == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]; then
                expertise="logn"
            elif [ "${i}" == "mistralai/Ministral-8B-Instruct-2410" ]; then
                expertise="linear"
            elif [ "${i}" == "deepseek-ai/deepseek-coder-7b-instruct-v1.5" ]; then
                expertise="nlogn"
            elif [ "${i}" == "mistralai/Mistral-7B-Instruct-v0.3" ]; then
                expertise="quadratic"
            elif [ "${i}" == "google/codegemma-7b-it" ]; then
                expertise="cubic"
            elif [ "${i}" == "codellama/CodeLlama-7b-Instruct-hf" ]; then
                expertise="exponential"
            fi
        fi

        model_name=$(echo "$i" | tr '/' '_')
        log_file="../logs/${GENERATION_TYPE}/${DATA_OPTION}/${OPTION}/${EXPER_REGENERATION}/${model_name}_${ONE_ROLE_DATA}_${expertise}.log"
        mkdir -p "../logs/${GENERATION_TYPE}/${DATA_OPTION}/${OPTION}/${EXPER_REGENERATION}"
        echo "Running: python3 ../code/sh_model.py --model $i ${@:2} --lang $ONE_ROLE_DATA --option $OPTION --data_option $DATA_OPTION --expertise $expertise --expertise_regeneration $EXPER_REGENERATION --generation_type $GENERATION_TYPE --tag $TAG --reuse $REUSE -> $log_file"
        python3 ../code/sh_model.py\ 
        --model "$i" "${@:2}" \
        --lang "$ONE_ROLE_DATA" \
        --option "$OPTION" \
        --data_option "$DATA_OPTION" \
        --expertise "$expertise" \
        --expertise_regeneration "$EXPER_REGENERATION" \
        --generation_type "$GENERATION_TYPE" \
        --tag "$TAG" \
        --reuse "$REUSE" > "$log_file"
    done
fi
