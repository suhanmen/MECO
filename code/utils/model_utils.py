import argparse
import torch
import pdb
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline
)
import copy

def setting_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, dest='model', help="LLM for inference")
    parser.add_argument('-p', '--pretrained', action='store_true')
    parser.add_argument('-l', '--lang', required=True, choices=['java','python'])
    parser.add_argument('-o', '--option', required=True, choices=[
        'single','single-codecomplex-expertise','single-codecomplex-expertise-with-confidence',
        'single-fewshot','multi-codecomplex-expertise','single-no-expertise_multi-no-expertise',
        'single-expertise_multi-no-expertise','single-no-expertise_multi-expertise',
        'multi-codecomplex-expertise-with-logit','single-no-expertise_multi-no-expertise-with-logit',
        'single-no-expertise_multi-expertise-with-logit','single-expertise_multi-no-expertise-with-logit',
        'judge_model-Ns_Nm','judge_model-Ns_Ym','judge_model-Ys_Nm','judge_model-Ys_Ym',
        'Round2','Round3','Round4','Round5',
        
        'REC-initialize','REC-discussion1', 'REC-discussion2', 'REC-discussion3',
        
        'CMD-Group1','CMD-Group2',
        "CMD-Group1_discussion1", "CMD-Group2_discussion1", 
        "CMD-Group1_discussion2", "CMD-Group2_discussion2",
        "CMD-tie",
        
        "MAD-affirmative_discussion1", "MAD-negative_discussion1",
        "MAD-affirmative_discussion2", "MAD-negative_discussion2",
        "MAD-affirmative_discussion3", "MAD-negative_discussion3",
        "MAD-affirmative_discussion4", "MAD-negative_discussion4",
        "MAD-affirmative_discussion5", "MAD-negative_discussion5",
        "MAD-judge1", "MAD-judge2", "MAD-judge3", "MAD-judge4", "MAD-judge5",
    ])
    parser.add_argument('-d', '--data_option', required=True, choices=[
        'full','step1-10%','step1-20%','step1-30%','step2,3'
    ])
    parser.add_argument('-e', '--expertise', choices=[
        'constant','logn','linear','nlogn','quadratic','cubic','exponential', 'None-expertise'
    ])
    parser.add_argument('-gtype','--generation_type', default='pipeline', choices=['pipeline','model'])
    parser.add_argument('-eptype','--expertise_regeneration', default='no_regen', choices=['no_regen','regen'])
    parser.add_argument('-tag','--tag', default='example')
    parser.add_argument('-reuse','--reuse', default='False', choices=['True','False'])
 
    args = parser.parse_args()
    # dtype and precision
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_impl = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_impl = 'sdpa'

    return {
        'compute_dtype': compute_dtype,
        'attn_implementation': attn_impl,
        'model_id': args.model,
        'pretrained': args.pretrained,
        'model_name': args.model.split('/')[-1],
        'dataset_path': 'codecomplex-simple',
        'output_dir': f"./{args.model.split('/')[-1]}-codecomplex-simple",
        'save_dir': f"./trained/{args.model.split('/')[-1]}-codecomplex-simple",
        'max_length': 4096,
        'input_max_length': 3584,
        'output_max_length': 512,
        'language': args.lang,
        'option': args.option,
        'data_option': args.data_option,
        'expertise': args.expertise,
        'generation_type': args.generation_type,
        'expertise_regeneration': args.expertise_regeneration,
        'tag': args.tag,
        'reuse': args.reuse
    }


def load_token_model(use_model: str, max_length: int, compute_dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        use_model, trust_remote_code=True, use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype
    )
    model = AutoModelForCausalLM.from_pretrained(
        use_model, quantization_config=bnb,
        torch_dtype=compute_dtype, device_map='auto', low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    return tokenizer, model


# def template_dataset(examples, tokenizer):
#     return {'text': tokenizer.apply_chat_template(
#         examples['messages'], tokenize=False, add_generation_prompt=True
#     )}

def template_dataset(examples, tokenizer):
    # Check if tokenizer supports system role in chat template
    supports_system = False
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            if "system" in tokenizer.chat_template:
                supports_system = True
        except:
            supports_system = False

    messages = examples['messages']
    if not supports_system:
        filtered_messages = []
        for m in messages:
            if m['role'] == 'system':
                filtered_messages.append({'role': 'user', 'content': m['content']})
            else:
                filtered_messages.append(m)
        messages = filtered_messages

    try:
        return {
            'text': tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        }
    except Exception as e:
        if "system" in str(e).lower():
            messages = [m for m in messages if m['role'] != 'system']
            return {
                'text': tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            }
        else:
            raise e

def truncate_user_message(messages, tokenizer, max_length, truncation_side='right'):
    # messages: list of dicts with 'role' and 'content'
    original_messages = copy.deepcopy(messages)
    
    for idx, m in enumerate(messages):
        system_text = m[0]['messages'][0]['content']
        user_text = m[0]['messages'][1]['content']
        system_tokens = tokenizer.encode(system_text)
        user_tokens = tokenizer.encode(user_text)
        if len(system_tokens)+len(user_tokens) > max_length: 
            if truncation_side == 'right':
                user_tokens = user_tokens[:max_length-len(system_tokens)-20] # 20 is special token
            else:
                user_tokens = user_tokens[-max_length+len(system_tokens)+20:] # 20 is special token
            truncated_text = tokenizer.decode(user_tokens)
            messages[idx][0]['messages'][1]['content'] = truncated_text
            print(f"Truncated user message: {idx}")
        else:
            pass
    return messages

def check_input_length(tokenizer, formatting_data, input_max_length: int):
    for idx, entry in enumerate(formatting_data):
        text = template_dataset(entry[0], tokenizer)['text']
        if len(tokenizer.encode(text)) > input_max_length:
            print(f"{idx} too long: {len(tokenizer.encode(text))}")
    print()
    
    
def save_generation_metadata(path: str, generated_text: str, generated_token_count: int, logit_score: float = None):
    data = {
        "generated_text": generated_text,
        "generated_token_count": generated_token_count
    }
    if logit_score is not None:
        data["logit_score"] = logit_score

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
