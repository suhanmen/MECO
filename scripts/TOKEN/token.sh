#!/bin/bash
##############################################################################################
LANGUAGES=("java" "python")
# Default

CATEGORIES=${1:-MECO} # You can chose mode
# Component: ONE_ROLE_PER_SETTING, MECO, CMD, REC, MAD

BOTH_INPUT_AND_OUTPUT="True" # input and output both
#Component: True or False

BOTH_INPUT_AND_OUTPUT_BASE="_BOTH_INPUT_AND_OUTPUT"
# Default

INPUT_DIR_BASE="../code/output_initialize/pipeline"
# Default

SAVE_PATH_BASE="../code/output_tokens"
# Default

EXPERTISE=("")
##############################################################################################

for CATEGORY in "${CATEGORIES[@]}"
do
    echo "Processing $CATEGORY..."
    
    if [ "$CATEGORY" = "ONE_ROLE_PER_SETTING" ]; then
        if [ "$BOTH_INPUT_AND_OUTPUT" = "True" ]; then
            CATEGORY_INPUT_AND_OUTPUT="${CATEGORY}${BOTH_INPUT_AND_OUTPUT_BASE}"
            echo "  Processing $CATEGORY_INPUT_AND_OUTPUT for both input and output analysis..."
            MODEL_DIRS=("Qwen2.5-Coder-7B-Instruct" "Mistral-7B-Instruct-v0.3" "Ministral-8B-Instruct-2410" "deepseek-coder-7b-instruct-v1.5" "Meta-Llama-3.1-8B-Instruct" "codegemma-7b-it" "CodeLlama-7b-Instruct-hf")
            #INPUT_DIR_BASE="../code/output_initialize/pipeline/SINGLE/single/step2,3/${LANG}/None-expertise"
            for MODEL_DIR in "${MODEL_DIRS[@]}"; do
                INPUT_DIRS=()
                for LANG in "${LANGUAGES[@]}"; do
                    INPUT_DIRS+=("${INPUT_DIR_BASE}/${CATEGORY}/single/step2,3/${LANG}/None-expertise/$(basename "${MODEL_DIR}")-codecomplex-simple")
                done
                SAVE_PATH="${SAVE_PATH_BASE}/${CATEGORY}/$(basename "${MODEL_DIR}")_both_input_and_output.txt"
                mkdir -p "$(dirname "$SAVE_PATH")"
                python ../code/utils_evaluation/token_both_input_and_output.py --inputdir "${INPUT_DIRS[@]}" --save_path "$SAVE_PATH"
            done
        
        else
            echo "  Processing $CATEGORY for combined analysis..."
            # Get all unique models from both java and python
            ALL_MODELS=()
            for LANG in "${LANGUAGES[@]}"; do
                ROOT_BASE="${INPUT_DIR_BASE}/${CATEGORY}/single/step2,3/${LANG}/None-expertise"
                if [ -d "$ROOT_BASE" ]; then
                    for MODEL in $(ls "$ROOT_BASE"); do
                        if [[ ! " ${ALL_MODELS[@]} " =~ " ${MODEL} " ]]; then
                            ALL_MODELS+=("$MODEL")
                        fi
                    done
                fi
            done
            
            # Process each model for combined (java + python)
            for MODEL in "${ALL_MODELS[@]}"; do
                echo "    Processing model: $MODEL"
                
                # Collect model directories from both java and python
                MODEL_DIRS=()
                for LANG in "${LANGUAGES[@]}"; do
                    ROOT_BASE="${INPUT_DIR_BASE}/${CATEGORY}/single/step2,3/${LANG}/None-expertise"
                    MODEL_DIR="$ROOT_BASE/$MODEL"
                    if [ -d "$MODEL_DIR" ]; then
                        MODEL_DIRS+=("$MODEL_DIR")
                    fi
                done
                
                SAVE_PATH="${SAVE_PATH_BASE}/${CATEGORY}/${basename "${MODEL}"}_combined.txt"
                mkdir -p "$(dirname "$SAVE_PATH")"
                python ../code/utils_evaluation/token_output.py --root_dirs "${MODEL_DIRS[@]}" --category "$CATEGORY" --save_path "$SAVE_PATH"
            done
        fi
        

    else
        if [ "$CATEGORY" = "CMD" ]; then
            SUBFOLDERS=("CMD-tie" "CMD-Group1" "CMD-Group2" "CMD-Group1_discussion1" "CMD-Group2_discussion1" "CMD-Group1_discussion2" "CMD-Group2_discussion2")
        elif [ "$CATEGORY" = "REC" ]; then
            SUBFOLDERS=("REC-discussion1" "REC-discussion2" "REC-initialize")
        elif [ "$CATEGORY" = "MECO" ]; then
            SUBFOLDERS=("single-codecomplex-expertise"  "multi-codecomplex-expertise")
        else
            echo "Unknown category: $CATEGORY, skipping..."
            continue
        fi

        if [ "$BOTH_INPUT_AND_OUTPUT" = "True" ]; then
            CATEGORY_INPUT_AND_OUTPUT="${CATEGORY}${BOTH_INPUT_AND_OUTPUT_BASE}"
            echo "  Processing $CATEGORY_INPUT_AND_OUTPUT for both input and output analysis..."    
            SAVE_PATH="${SAVE_PATH_BASE}/${CATEGORY}/combined_both_input_and_output.txt"
            mkdir -p "$(dirname "$SAVE_PATH")"
            INPUT_DIRS=()
            for SUBFOLDER in "${SUBFOLDERS[@]}"; do
                for LANG in "${LANGUAGES[@]}"; do
                    ROOT_DIR="${INPUT_DIR_BASE}/${CATEGORY}/${SUBFOLDER}/step2,3/${LANG}"
                    if [ -d "$ROOT_DIR" ]; then
                        if [ "$CATEGORY" = "MECO" ]; then
                            for EXPERTISE_DIR in "$ROOT_DIR"/*; do
                                if [ -d "$EXPERTISE_DIR" ]; then
                                    for MODEL_DIR in "$EXPERTISE_DIR"/*; do
                                        if [ -d "$MODEL_DIR" ]; then
                                            INPUT_DIRS+=("$MODEL_DIR")
                                        fi
                                    done
                                fi
                            done
                        else
                            for MODEL_DIR in "$ROOT_DIR"/*; do
                                if [ -d "$MODEL_DIR" ]; then
                                    INPUT_DIRS+=("$MODEL_DIR")
                                fi
                            done
                        fi
                    fi
                done
            done  
            python ../code/utils_evaluation/token_both_input_and_output.py --inputdir "${INPUT_DIRS[@]}" --save_path "$SAVE_PATH"
        
        else
            echo "  Processing $CATEGORY for combined analysis..."
            SAVE_PATH="${SAVE_PATH_BASE}/${CATEGORY}/combined.txt"
            mkdir -p "$(dirname "$SAVE_PATH")"
            MODEL_DIRS=()
            for SUBFOLDER in "${SUBFOLDERS[@]}"; do
                for LANG in "${LANGUAGES[@]}"; do
                    ROOT_DIR="${INPUT_DIR_BASE}/${CATEGORY}/${SUBFOLDER}/step2,3/${LANG}"
                    if [ -d "$ROOT_DIR" ]; then
                        if [ "$CATEGORY" = "MECO" ]; then
                            for EXPERTISE_DIR in "$ROOT_DIR"/*; do
                                if [ -d "$EXPERTISE_DIR" ]; then
                                    for MODEL_DIR in "$EXPERTISE_DIR"/*; do
                                        if [ -d "$MODEL_DIR" ]; then
                                            MODEL_DIRS+=("$MODEL_DIR")
                                        fi
                                    done
                                fi
                            done
                        else
                            for MODEL_DIR in "$ROOT_DIR"/*; do
                                if [ -d "$MODEL_DIR" ]; then
                                    MODEL_DIRS+=("$MODEL_DIR")
                                fi
                            done
                        fi
                    fi
                done
            done  
            echo MODEL_DIRS: "${MODEL_DIRS[@]}"
            python ../code/utils_evaluation/token_output.py --root_dirs "${MODEL_DIRS[@]}" --category "$CATEGORY" --save_path "$SAVE_PATH"
        fi   
    fi
done