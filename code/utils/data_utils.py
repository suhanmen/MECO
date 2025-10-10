import json
from re import T
from utils_Instruction.MECO import get_prompt_messages as meco_get
from utils_Instruction.RECONCILE import get_prompt_messages as rec_get
from utils_Instruction.CMD import get_prompt_messages as cmd_get
from utils_Instruction.MAD import get_prompt_messages as mad_get
from utils.format_utils import model_name_format, back_name_format, label_format
import os
import pdb

def data_load_format(dataset_path, language, model_name, option, data_option,
                     generation_type, expertise_regeneration, TAG, expertise=None):
    # 1) read original data
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]

    mad_skip_pass_idx = set()  
    vote = None
    vote_path = None
    
    vote_opts = [
        'multi-codecomplex-expertise', 'Round2', 'Round3', 'Round4', 'Round5',
        'single-expertise_multi-no-expertise', 'single-no-expertise_multi-expertise',
        'single-no-expertise_multi-no-expertise',
        'multi-codecomplex-expertise-with-logit',
        'single-no-expertise_multi-no-expertise-with-logit',
        'single-no-expertise_multi-expertise-with-logit',
        'single-expertise_multi-no-expertise-with-logit',
        'judge_model-Ns_Nm','judge_model-Ns_Ym','judge_model-Ys_Nm','judge_model-Ys_Ym',
        
        'REC-discussion1','REC-discussion2',
        
        "CMD-Group1_discussion1", "CMD-Group2_discussion1", 
        "CMD-Group1_discussion2", "CMD-Group2_discussion2",
        'CMD-tie',
        
        "MAD-negative_discussion1",
        "MAD-affirmative_discussion2", "MAD-negative_discussion2",
        "MAD-affirmative_discussion3", "MAD-negative_discussion3",
        "MAD-affirmative_discussion4", "MAD-negative_discussion4",
        "MAD-affirmative_discussion5", "MAD-negative_discussion5",
        "MAD-judge1", "MAD-judge2", "MAD-judge3", "MAD-judge4", "MAD-judge5",
    ]


#######################################################################################################
#    # 2) data option according to the path
    VOTING_BASE_PATH = "../code/output_voting/pipeline"
    
    if option in vote_opts:
        if option.startswith('Round'):
            prev = int(option.replace('Round', '')) - 1
            vote_path = f"{VOTING_BASE_PATH}/MECO/Round{prev}/{data_option}/output-{language}-voting.json"


        elif option.startswith('REC'):
            if option == 'REC-discussion1':
                vote_path = f"{VOTING_BASE_PATH}/REC/REC-initialize/{data_option}/output-{language}-voting.json"
            else:
                prev = int(option.replace('REC-discussion', '')) - 1
                vote_path = f"{VOTING_BASE_PATH}/REC/REC-discussion{prev}/{data_option}/output-{language}-voting.json"


        elif option.startswith('CMD'):
            if option == "CMD-tie":
                this_vote_path = f"{VOTING_BASE_PATH}/CMD/CMD-Group1_discussion2/{data_option}/output-{language}-voting.json"
                other_vote_path = f"{VOTING_BASE_PATH}/CMD/CMD-Group2_discussion2/{data_option}/output-{language}-voting.json"
            else:
                base_group = option.split('_')[0]  # CMD-Group1 or CMD-Group2
                discussion_round = option.split('_')[1]  # discussion1, discussion2, ...
                group_num = base_group.split('-')[1]  # Group1 or Group2
                other_group = 'CMD-Group2' if group_num == 'Group1' else 'CMD-Group1'

                if discussion_round == 'discussion1':
                    this_vote_dir = base_group
                    other_vote_dir = other_group
                else:
                    prev_round_num = int(discussion_round.replace("discussion", "")) - 1
                    this_vote_dir = f"{base_group}_discussion{prev_round_num}"
                    other_vote_dir = f"{other_group}_discussion{prev_round_num}"

                this_vote_path = f"{VOTING_BASE_PATH}/CMD/{this_vote_dir}/{data_option}/output-{language}-voting.json"
                other_vote_path = f"{VOTING_BASE_PATH}/CMD/{other_vote_dir}/{data_option}/output-{language}-voting.json"

            try:
                with open(this_vote_path, 'r', encoding='utf-8') as vf1:
                    this_vote = json.load(vf1)
                with open(other_vote_path, 'r', encoding='utf-8') as vf2:
                    other_vote = json.load(vf2)
                vote = {"this_group": this_vote, "other_group": other_vote}
            except FileNotFoundError:
                vote = None

        elif option.startswith("MAD"):
            if "discussion" in option:
                round_num = int(option.split('_')[-1].replace("discussion", ""))
                is_affirmative = "affirmative" in option

                if round_num == 1 and not is_affirmative:
                    opposite_vote_option = "MAD-affirmative_discussion1"
                elif round_num > 1:
                    prev_round = round_num - 1 if is_affirmative else round_num
                    opposite_role = "negative" if is_affirmative else "affirmative"
                    opposite_vote_option = f"MAD-{opposite_role}_discussion{prev_round}"

                if (round_num == 1 and not is_affirmative) or round_num > 1:
                    vote_path = f"{VOTING_BASE_PATH}/MAD/{opposite_vote_option}/{data_option}/output-{language}-voting.json"
                    try:
                        with open(vote_path, 'r', encoding='utf-8') as jf:
                            vote = json.load(jf)
                    except FileNotFoundError:
                        vote = None

            elif option.startswith("MAD-judge"):
                round_num = int(option.replace("MAD-judge", ""))
                if round_num > 1:
                    judge_option = f"MAD-judge{round_num - 1}"
                    judge_vote_path = f"{VOTING_BASE_PATH}/MAD/{judge_option}/{data_option}/output-{language}-voting.json"
                    try:
                        with open(judge_vote_path, 'r', encoding='utf-8') as jf:
                            judge_results = json.load(jf)
                            for idx_str, result in judge_results.items():
                                if isinstance(result, dict) and result['qwen'].get("pass", 0) == 1:
                                    mad_skip_pass_idx.add(int(idx_str))
                    except FileNotFoundError:
                        pass
        
        elif option.startswith('judge_model'):
            if option == 'judge_model-Ns_Nm':
                vote_path = f"{VOTING_BASE_PATH}/MECO/Round3/pipeline/judge_model-Ns_Nm/{data_option}/output-{language}-voting.json"
            elif option == 'judge_model-Ns_Ym':
                vote_path = f"{VOTING_BASE_PATH}/MECO/Round3/pipeline/judge_model-Ns_Ym/{data_option}/output-{language}-voting.json"
            elif option == 'judge_model-Ys_Nm':
                vote_path = f"{VOTING_BASE_PATH}/MECO/Round3/pipeline/judge_model-Ys_Nm/{data_option}/output-{language}-voting.json"
            elif option == 'judge_model-Ys_Ym':
                vote_path = f"{VOTING_BASE_PATH}/MECO/Round3/pipeline/judge_model-Ys_Ym/{data_option}/output-{language}-voting.json"
        
        else:
            vote_path = f"{VOTING_BASE_PATH}/{TAG}/single-codecomplex-expertise/{data_option}/output-{language}-voting.json"
            # logit score
            #vote_path = f"./scripts/result/0.voting/{data_option}/pipeline/{option}/output-{language}-voting.json"

           
            

#######################################################################################################
    if vote_path and vote is None:
        try:
            with open(vote_path, 'r', encoding='utf-8') as vf:
                vote = json.load(vf)
        except FileNotFoundError:
            vote = None

    formatting_data = []
    idx_list = []
    
    cmd_not_tie_list = [] 
    tie_idx_list = [] 
    
    for idx, example in enumerate(data):
        src = example.get("src", "")

        kwargs = {
            "language": language,
            "expertise": expertise,
            "data_option": data_option
        }

        # MECO - multi-codecomplex-expertise
        if vote and expertise and option == 'multi-codecomplex-expertise':
            expert_key = f"{expertise}-{back_name_format(model_name)}"
            try:
                pred = vote[str(idx)][expert_key]['predicted_complexity']
                if label_format(pred) == expertise:
                    idx_list.append(idx)
            except KeyError:
                pass

        # CMD
        if option.startswith("CMD"):
            
            if option == "CMD-tie" and vote:
                this_group_vote = vote["this_group"].get(str(idx), {}).get("vote", [])
                other_group_vote = vote["other_group"].get(str(idx), {}).get("vote", [])

                if not this_group_vote or not other_group_vote:
                    continue  # skip if there is no vote result in the index

                total_vote = [a + b for a, b in zip(this_group_vote, other_group_vote)]
                
                if sum(total_vote) == 0:
                    cmd_not_tie_list.append({
                        "idx": idx,
                        'predicted_complexity' : -1,
                        "str_label": None
                    })
                    continue
                
                max_vote = max(total_vote)
                max_indices = [i for i, v in enumerate(total_vote) if v == max_vote]

                if len(max_indices) > 1:
                    tie_idx_list.append(idx)
                    candidates = []
                    for comp_idx in max_indices:
                        explanations = []
                        for group_name in ["this_group", "other_group"]:
                            vote_entry = vote[group_name].get(str(idx), {})
                            for model_key, out in vote_entry.items():
                                if model_key == "vote":
                                    continue
                                if out["predicted_complexity"] == comp_idx:
                                    explanations.append(out.get("explanation", ""))
                        candidates.append({
                            "complexity": label_format(comp_idx),
                            "explanations": explanations
                        })
                    kwargs["candidates"] = candidates
    
                else:            
                    maj_class = max_indices[0]
                    cmd_not_tie_list.append({
                    "idx": idx,
                    'predicted_complexity' : maj_class,
                    "str_label": label_format(maj_class)
                    })
                    
                    continue
                    
                    

        
            elif isinstance(vote, dict) and "this_group" in vote:
                # CMD group discussion
                in_group_data = []
                out_group_data = []
                this_group = vote["this_group"].get(str(idx), {})
                other_group = vote["other_group"].get(str(idx), {})

                for out in this_group.values():
                    if isinstance(out, dict):
                        in_group_data.append({
                            "complexity": label_format(out['predicted_complexity']),
                            "confidence": out.get('confidence'),
                            "explanation": out.get('explanation', "")
                        })

                for out in other_group.values():
                    if isinstance(out, dict):
                        out_group_data.append({
                            "complexity": label_format(out['predicted_complexity'])
                        })

                kwargs["in_group_data"] = in_group_data
                kwargs["out_group_data"] = out_group_data

        # MAD
        elif option.startswith("MAD"):
            if "discussion" in option and vote:
                # MAD vote_sentence
                entries = []
                for model_key, out in vote.get(str(idx), {}).items():
                    if model_key == 'vote': continue
                    name_ = model_name_format(model_key.split('-')[-1])
                    comp_ = label_format(out['predicted_complexity'])
                    expl_ = out.get('explanation', "")
                    entry = {
                        "model_name": name_,
                        "complexity": comp_,
                        "explanation": expl_
                    }
                    entries.append(entry)

                # for previous_negative_output, previous_affirmative_output
                vote_sentence = json.dumps(entries, ensure_ascii=False, indent=2)
                if "affirmative" in option:
                    kwargs["previous_negative_output"] = vote_sentence
                else:
                    kwargs["previous_affirmative_output"] = vote_sentence

            elif option.startswith("MAD-judge"):
                judge_round = int(option.replace("MAD-judge", ""))
                for side in ["affirmative", "negative"]:
                    vote_path = f"{VOTING_BASE_PATH}/MAD-{side}_discussion{judge_round}/{data_option}/output-{language}-voting.json"
                    try:
                        with open(vote_path, 'r', encoding='utf-8') as vf:
                            vote_dict = json.load(vf).get(str(idx), {})
                        entries = []
                        for model_key, out in vote_dict.items():
                            if model_key == 'vote': continue
                            name_ = model_name_format(model_key.split('-')[-1])
                            comp_ = label_format(out['predicted_complexity'])
                            expl_ = out.get('explanation', "")
                            entry = {
                                "model_name": name_,
                                "complexity": comp_,
                                "explanation": expl_
                            }
                            entries.append(entry)
                        vote_sentence = json.dumps(entries, ensure_ascii=False, indent=2)
                        kwargs[f"{side}_vote_sentence"] = vote_sentence
                    except FileNotFoundError:
                        kwargs[f"{side}_vote_sentence"] = ""
        
        
        # Base, REC
        elif vote:
            # general vote_sentence
            entries = []
            for model_key, out in vote.get(str(idx), {}).items():
                if model_key == 'vote': continue
                name_ = model_name_format(model_key.split('-')[-1])
                exp_  = model_key.split('-')[0] if '-' in model_key else None
                comp_ = label_format(out['predicted_complexity'])
                expl_ = out.get('explanation', "")
                conf_ = out.get('confidence')
                entry = {
                    "model_name": name_,
                    **({"expertise": exp_} if exp_ else {}),
                    "complexity": comp_,
                    **({"confidence": conf_} if conf_ is not None else {}),
                    "explanation": expl_
                }
                entries.append(entry)
            vote_sentence = json.dumps(entries, ensure_ascii=False, indent=2)
            kwargs["vote_sentence"] = vote_sentence


####################################################################################################### 
        # 5) call prompt generation function
        if option.startswith("REC"):
            messages = rec_get(option, src, **kwargs)
        elif option.startswith("CMD"):
            messages = cmd_get(option, src, **kwargs)
        elif option.startswith("MAD"):
            messages = mad_get(option, src, **kwargs)
        else:
            messages = meco_get(option, src, **kwargs)

        formatting_data.append([{"messages": messages}])
        
    if option == "CMD-tie":
        # save not tie result
        save_path = f"{VOTING_BASE_PATH}/CMD/CMD-tie/{data_option}/CMD-Not-tie_list-{language}.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(cmd_not_tie_list, f, ensure_ascii=False, indent=2)

        # save tie indices
        tie_save_path = f"{VOTING_BASE_PATH}/CMD/CMD-tie/{data_option}/CMD-Tie-indices-{language}.json"
        with open(tie_save_path, 'w', encoding='utf-8') as f:
            json.dump(tie_idx_list, f, ensure_ascii=False, indent=2)
    
    return formatting_data, idx_list, mad_skip_pass_idx

