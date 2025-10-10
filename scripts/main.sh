METHOD=${1:-MECO}
GPU_ID=${2:-0}
# Components: MECO, CMD, MAD, RECONCILE

if [ "${METHOD}" == "Single" ]; then
    # Single MECO setting
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single pipeline
    bash ./MECO/score.sh MECO step2,3 single
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single-no-expertise_multi-no-expertise pipeline
    bash ./MECO/score.sh MECO step2,3 single-no-expertise_multi-no-expertise

    # Single fewshot setting
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single-fewshot pipeline
    bash ./MECO/score.sh MECO step2,3 single-fewshot

elif [ "${METHOD}" == "Find_expertise" ]; then
    # Find expertise (This make expertise json file)
    sh ./MECO/scripts.sh ${GPU_ID} MECO step1-10% single pipeline
    bash ./MECO/score.sh MECO step1-10% single
    sh ./MECO/scripts.sh ${GPU_ID} MECO step1-20% single pipeline
    bash ./MECO/score.sh MECO step1-20% single    
    sh ./MECO/scripts.sh ${GPU_ID} MECO step1-30% single pipeline
    bash ./MECO/score.sh MECO step1-30% single

elif [ "${METHOD}" == "MECO" ]; then
    # Expertise generation
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single-codecomplex-expertise pipeline
    bash ./MECO/score.sh MECO step2,3 single-codecomplex-expertise
    echo "================== Expertise generation done =================="

    # Make logit score
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 multi-codecomplex-expertise model
    echo "================== Make logit score done =================="

    ## Debate    
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 multi-codecomplex-expertise pipeline
    bash ./MECO/score.sh MECO step2,3 multi-codecomplex-expertise

    ## Calculate token
    sh ./TOKEN/token.sh ${METHOD}


elif [ "${METHOD}" == "MECO_Multi_Round" ]; then
    # This setting only use finished MECO setting.
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 Round3 pipeline
    bash ./MECO/score.sh MECO step2,3 Round3
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 Round4 pipeline
    bash ./MECO/score.sh MECO step2,3 Round4
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 Round5 pipeline
    bash ./MECO/score.sh MECO step2,3 Round5
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 Round6 pipeline
    bash ./MECO/score.sh MECO step2,3 Round6


elif [ "${METHOD}" == "MECO_ONE_ROLE_PER_SETTING" ]; then
    # This setting only use finished MECO setting.
    # Single no expertise setting
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single-no-expertise_multi-expertise pipeline
    bash ./MECO/score.sh MECO step2,3 single-no-expertise_multi-expertise

    # Multi no expertise setting
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single-expertise_multi-no-expertise pipeline
    bash ./MECO/score.sh MECO step2,3 single-expertise_multi-no-expertise


elif [ "${METHOD}" == "MECO_JUDGE_MODEL" ]; then
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 judge_model-Ns_Nm pipeline "Qwen/Qwen2.5-Coder-7B-Instruct"
    bash ./MECO/score.sh MECO step2,3 judge_model-Ns_Nm
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 judge_model-Ys_Nm pipeline "Qwen/Qwen2.5-Coder-7B-Instruct"
    bash ./MECO/score.sh MECO step2,3 judge_model-Ys_Nm
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 judge_model-Ns_Ym pipeline "Qwen/Qwen2.5-Coder-7B-Instruct"
    bash ./MECO/score.sh MECO step2,3 judge_model-Ns_Ym
    # YS_YM is MECO setting


elif [ "${METHOD}" == "MECO_LOGIT" ]; then
    # This setting only use finished MECO, MECO_ONE_ROLE_PER_SETTING setting.
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single-no-expertise_multi-no-expertise-with-logit model
    bash ./MECO/score.sh MECO step2,3 single-no-expertise_multi-no-expertise-with-logit
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single-no-expertise_multi-expertise-with-logit model
    bash ./MECO/score.sh MECO step2,3 single-no-expertise_multi-expertise-with-logit
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single-expertise_multi-no-expertise-with-logit
    bash ./MECO/score.sh MECO step2,3 single-expertise_multi-no-expertise-with-logit
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 multi-codecomplex-expertise-with-logit model
    bash ./MECO/score.sh MECO step2,3 multi-codecomplex-expertise-with-logit


elif [ "${METHOD}" == "MECO_CONFIDENCE" ]; then
    sh ./MECO/scripts.sh ${GPU_ID} MECO step2,3 single-codecomplex-expertise-with-confidence model
    bash ./MECO/score.sh MECO step2,3 single-codecomplex-expertise-with-confidence
    

elif [ "${METHOD}" == "CMD" ]; then
    sh ./CMD/CMD-initialize.sh ${GPU_ID}
    sh ./CMD/CMD-discussion.sh ${GPU_ID}
    sh ./CMD/CMD-tie.sh ${GPU_ID}
    sh ./TOKEN/token.sh ${METHOD}


elif [ "${METHOD}" == "MAD" ]; then
    sh ./MAD/MAD-initialize.sh ${GPU_ID}
    sh ./TOKEN/token.sh ${METHOD}


elif [ "${METHOD}" == "RECONCILE" ]; then
    #sh ./RECONCILE/REC-initialize.sh ${GPU_ID}
    sh ./RECONCILE/REC-discussion.sh ${GPU_ID}
    sh ./TOKEN/token.sh ${METHOD}
fi