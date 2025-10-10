import os
import json
import shutil
import torch
import warnings
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import pdb

from utils.model_utils import (
    setting_data,
    load_token_model,
    template_dataset,
    truncate_user_message,
    check_input_length
)
from utils.data_utils import data_load_format
from utils.model_utils import save_generation_metadata
from utils.format_utils import load_MARKER
import torch.nn.functional as F

def main():
    cfg = setting_data() # Parse CLI args and confi
    initialize_data_path = '../data' # Determine dataset paths
    
    if cfg['data_option'] == 'full': # Load full dataset
        paths = {
            'java': f'{initialize_data_path}/full/java_data.jsonl',
            'python': f'{initialize_data_path}/full/python_data.jsonl'
        }
        
    elif cfg['data_option'].startswith('step1'): # For finding time complexity expertise
        paths = {
            'java': f"{initialize_data_path}/codecomplex_data_sampling/{cfg['data_option']}-professional/java-multi_test_data.jsonl",
            'python': f"{initialize_data_path}/codecomplex_data_sampling/{cfg['data_option']}-professional/python-multi_test_data.jsonl"
        }
        
    else:  # step2,3 # For evaluation
        paths = {
            'java': f"{initialize_data_path}/codecomplex_data_sampling/step2,3-490/java-multi_test_data.jsonl",
            'python': f"{initialize_data_path}/codecomplex_data_sampling/step2,3-490/python-multi_test_data.jsonl"
        }

    dataset_path = paths[cfg['language']]
    
    # Load and format data
    formatting_data, idx_list, mad_skip_pass_idx = data_load_format(
        dataset_path,
        cfg['language'],
        cfg['model_name'],
        cfg['option'],
        cfg['data_option'],
        cfg['generation_type'],
        cfg['expertise_regeneration'],
        cfg['tag'],
        cfg['expertise']
    )


##############################################################################################
    # Prepare output directory
    initialize_output_path = f'../code/output_initialize'
    
    if cfg['expertise'] != None:
        output_path = os.path.join(
            initialize_output_path,
            cfg['generation_type'],
            cfg['tag'],
            cfg['option'],
            cfg['data_option'],
            cfg['language'],
            str(cfg['expertise']),
            f"{cfg['model_name']}-{cfg['dataset_path']}"
        )
    else:
        output_path = os.path.join(
            initialize_output_path,
            cfg['generation_type'],
            cfg['tag'],
            cfg['option'],
            cfg['data_option'],
            cfg['language'],
            f"{cfg['model_name']}-{cfg['dataset_path']}"
        )
        
    if cfg['expertise_regeneration'] == 'regen':
        output_path = os.path.join(output_path, 'regen')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    
##############################################################################################
# Load model & tokenizer
    tokenizer, model = load_token_model(
        cfg['model_id'], cfg['max_length'], cfg['compute_dtype']
    )

    # Setup generation pipeline or direct model
    if cfg['generation_type'] == 'pipeline':
        pipe = pipeline(
            'text-generation', 
            model=model, 
            tokenizer=tokenizer,
            torch_dtype=cfg['compute_dtype'], 
            device_map='auto'
        )
    else:
        model.config.pad_token_id = model.config.eos_token_id




##############################################################################################
# Run inference
    #formatting_data=formatting_data[:10]
    formatting_data = truncate_user_message(formatting_data, tokenizer, cfg['input_max_length'])
    check_input_length(tokenizer, formatting_data, cfg['input_max_length'])

    pbar = tqdm(total=len(formatting_data), desc="Testing")
    logit_list = {}
    generation_results = {}
    resume = True if cfg.get('reuse') == 'True' else False
    logit_file_path = os.path.join(output_path, "logit_score.json")
    
    if resume and cfg['generation_type'] == "model": # load logit_score.json if resume is True
        if os.path.exists(logit_file_path):
            with open(logit_file_path, 'r') as f:
                logit_list = json.load(f)
                print(f"Loaded existing logit data: {len(logit_list)} entries")
    
    for i, entry in enumerate(formatting_data):
        response_path = os.path.join(output_path, f"responce_{i:04d}.txt")
        
        if os.path.exists(response_path) and resume: # skip if response_path exists and resume is True
            print(f"Skip {i} because it already exists")
            pbar.update(1)
            continue
        
        if i in logit_list and resume and cfg['generation_type'] == "model":
            print(f"Skip {i} because it already exists")
            pbar.update(1)
            continue
        
        text = template_dataset(entry[0], tokenizer)['text']
        input_text = template_dataset(entry[0], tokenizer)['text']
        
        # Pipeline generation             
        if cfg['generation_type'] == 'pipeline':
            out = pipe(
                    text,
                    max_new_tokens=cfg['output_max_length'],
                    #max_length=cfg['max_length']-cfg['output_max_length'],
                    return_full_text=True,
                    padding=True,
                    truncation=True, # 
                    temperature= { 'CMD':0.25, 'MAD':0.1 }.get(cfg['option'].split('-')[0], 1.0)
                    )
            
            result = out[0]['generated_text']
            
            MARKER = load_MARKER(cfg['model_name'])
            if MARKER == "":
                warnings.warn("No marker found")
                print(cfg['model_name'])
            
            generated_only = result.rsplit(MARKER, 1)[-1]
            encoded = tokenizer(generated_only, add_special_tokens=False)
            
            generation_results[f"{i}"] = {
                "generated_text": generated_only,
                "generated_token_count": len(encoded['input_ids']) # Remove EQS tokens
            }
            print(result)
        ##############################################################################################
        # model generation
        else:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=cfg['input_max_length']).to(model.device)
            out = model.generate(**inputs, max_new_tokens=cfg['output_max_length'], return_dict_in_generate=True, output_scores=True, return_legacy_cache=True)
            result = tokenizer.decode(out.sequences[0],skip_special_tokens=False) 
            
            # Calculate logit
            generated_tokens = out.sequences[0][inputs['input_ids'].shape[-1]:]
            start_idx = tokenizer(' "complexity": "', add_special_tokens=False)["input_ids"]
            start_pattern = torch.tensor(start_idx, device="cuda")
            start_window_size = len(start_pattern) 
            start_unfolded = generated_tokens.unfold(0, start_window_size, 1)
            start_match_idx = (start_unfolded == start_pattern).all(dim=1).nonzero(as_tuple=True)[0]
            
            # Some models have different start_idx, so we need to check again
            if start_match_idx.shape[0] == 0:
                start_idx = tokenizer('"complexity": "', add_special_tokens=False)["input_ids"]
                start_pattern = torch.tensor(start_idx, device="cuda")
                start_window_size = len(start_pattern) 
                start_unfolded = generated_tokens.unfold(0, start_window_size, 1)
                start_match_idx = (start_unfolded == start_pattern).all(dim=1).nonzero(as_tuple=True)[0]
            
            if start_match_idx.shape[0] == 0:
                print("No match")
                final_logit_score = 0
            else:
                logit_idx = start_match_idx+start_window_size
                if logit_idx.shape[0] == 1:
                    logit_idx = logit_idx[0]    
                
                try:               
                    logit_score = out.scores[logit_idx]
                    prob_score = F.softmax(logit_score, dim=-1)
                    final_logit_score = max(prob_score[0]).item()
                except:
                    final_logit_score = 0
            
            logit_list[i] = {"generated_text" : tokenizer.decode(generated_tokens) ,"logit_score": final_logit_score}        
            
            with open(logit_file_path, 'w') as f:
                json.dump(logit_list, f, indent=4)
            
            print(result)
            print(f"logit_score : {final_logit_score}")


##############################################################################################
# Save or copy based on idx_list
        if cfg['option'].startswith('multi-codecomplex-expertise') and cfg['expertise_regeneration']=='no_regen':
            if i in idx_list and cfg['generation_type']=='pipeline':
                # copy from single-expertise baseline
                src_file = response_path.replace(cfg['option'], 'single-codecomplex-expertise')
                shutil.copy(src_file, response_path)
                pbar.update(1)
                continue

        with open(response_path, 'w') as f:
            if cfg['generation_type'] == 'pipeline':
                f.write(repr(out))
            else:
                result_format = [{'generated_text': result}]
                f.write(json.dumps(result_format))
            
        pbar.update(1)
    pbar.close()
    
    # Skip idx in MAD-judge round is copied from the previous judge round
    if cfg["option"].startswith("MAD-judge"):
        round_num = int(cfg["option"].replace("MAD-judge", ""))
        if round_num >= 2:
            # Set previous judge round path
            prev_option = f"MAD-judge{round_num - 1}"
            prev_resp_dir = output_path.replace(cfg["option"], prev_option)
            
            for i in mad_skip_pass_idx:
                src_file = os.path.join(prev_resp_dir, f"responce_{i:04d}.txt")
                response_path = os.path.join(output_path, f"responce_{i:04d}.txt")
                if os.path.exists(src_file):
                    shutil.copy(src_file, response_path)
                    print(f"[MAD SKIP] Copied from previous judge: {src_file} → {response_path}")
                else:
                    print(f"[WARNING] Could not find: {src_file} for skip index {i}")
    
    if cfg['generation_type'] == "model":
        with open(f"{output_path}/logit_score.json", 'w') as f:
            json.dump(logit_list, f, indent=4)
    
    if cfg['generation_type'] == "pipeline":
        gen_result_path = os.path.join(output_path, "generation_results.json")
        if os.path.exists(gen_result_path):
            with open(gen_result_path, 'r', encoding='utf-8') as f:
                try:
                    prev_results = json.load(f)
                except json.JSONDecodeError:
                    prev_results = {}
        else:
            prev_results = {}
        prev_results.update(generation_results)
        with open(gen_result_path, 'w', encoding='utf-8') as f:
            json.dump(prev_results, f, indent=4, ensure_ascii=False)
        

##############################################################################################
if __name__ == '__main__':
    main()
