from email import message_from_string
import argparse, ast, glob, os, sys
from collections import OrderedDict
import pdb
try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("pip install transforme!")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.format_utils import load_MARKER

# ──────────────────────────────────────────────────────────────────
def model_name_to_marker(input_dir: str):
    if 'Qwen2.5-Coder' in input_dir:
        return "Qwen/Qwen2.5-Coder-7B-Instruct"
    elif 'Mistral-7B-Instruct' in input_dir:
        return "mistralai/Mistral-7B-Instruct-v0.3"
    elif 'Ministral-8B-Instruct' in input_dir:
        return "mistralai/Ministral-8B-Instruct-2410"
    elif 'deepseek-coder' in input_dir:
        return "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    elif 'Meta-Llama-3.1' in input_dir:
        return "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif 'codegemma-7b-it' in input_dir:
        return "google/codegemma-7b-it"
    elif 'CodeLlama-7b-Instruct' in input_dir:
        return "codellama/CodeLlama-7b-Instruct-hf"

def load_tok(model_name: str):
    print(f"🔑  Loading tokenizer for '{model_name}' …")
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def n_tok(tok, txt):
    return len(tok(txt, add_special_tokens=False)["input_ids"])

def count_file(path: str, tok):
    MARKER = load_MARKER(path)
    
    if MARKER == "":
        pdb.set_trace()
    
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    data = ast.literal_eval(raw)

    full, after = 0, 0
    for item in data:
        if not isinstance(item, dict) or "generated_text" not in item:
            continue
        gtxt = item["generated_text"]
        #pdb.set_trace()
        full  += n_tok(tok, gtxt)
        
        if MARKER in gtxt:
            after += n_tok(tok, gtxt.rsplit(MARKER, 1)[-1])
    return full, after

# ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Calculate token statistics from multiple response_*.txt files")
    ap.add_argument("--inputdirs", nargs="+", required=True,help="Pass multiple directories separated by space")
    ap.add_argument("--model", required=False, help="Model name/path", default=None)
    ap.add_argument("--save_path", required=True, help="Save result file path")
    args = ap.parse_args()

    per_file = OrderedDict()        # Record statistics per file
    grand_total_full = grand_total_after = 0
    grand_n_files = 0
    
    # Variables for language-wise statistics
    java_total_full = java_total_after = 0
    java_n_files = 0
    python_total_full = python_total_after = 0
    python_n_files = 0
    
    for inputdir in args.inputdirs:
        model_name = model_name_to_marker(inputdir)
        tok = load_tok(model_name)
        #pdb.set_trace()
        files = sorted(glob.glob(os.path.join(inputdir, "responce_*.txt")))
        if not files:
            print(f"⚠️  {inputdir} has no responce_*.txt", file=sys.stderr)
            continue
        
        for fp in files:
            try:
                full, after = count_file(fp, tok)
                full = full if full <= 4096 else 4096 # 4096 is the max length of the model, the model will truncate the text itself
                after = after #if after < 513 else 512 # 512 is the max length of the model, the previous special token is 513
                
                if 'java' in inputdir:
                    lang = 'java'
                    java_total_full += full
                    java_total_after += after
                    java_n_files += 1
                else:
                    lang = 'python'
                    python_total_full += full
                    python_total_after += after
                    python_n_files += 1
                
                if "SINGLE" in args.inputdirs:
                    per_file[lang + "/" + os.path.basename(fp)] = (full, after)
                else:
                    per_file[lang + "/" + fp.split("pipeline")[-1].split("/")[2] + "/" + os.path.basename(fp)] = (full, after)
                
                #pdb.set_trace()
                grand_total_full += full 
                grand_total_after += after
                grand_n_files += 1
            except Exception as e:
                print(f"[Warning] {fp}: {e}", file=sys.stderr)
                
            

    # ─────── Write result message ──────────────────────────────────
    message = []
    message.append("\n🧮  Token count per file")
    message.append("-" * 60)
    message.append(f"{'filename':30} {'GEN_ALL':>10} {'GEN_AFTER':>12}")
    
    # Combine java and python parts into a single loop to remove duplicates
    for lang in ['java', 'python']:
        files = [(name, full, aft) for name, (full, aft) in per_file.items() if name.startswith(f'{lang}/')]
        if lang == 'python':
            message.append("-" * 60)
            message.append("-" * 60)
        prev_tag = None
        for name, full, aft in files:
            try:
                tag = name.split('/')[1]
            except IndexError:
                tag = None
            if tag != prev_tag:
                if prev_tag is not None:
                    message.append("-" * 60)
                    message.append("-" * 60)
                prev_tag = tag
            message.append(f"{name:30} {full:10,} {aft:12,}")
        if lang == 'java':
            # Java total and average
            java_avg_full = java_total_full / java_n_files if java_n_files else 0
            java_avg_after = java_total_after / java_n_files if java_n_files else 0
        else:
            # Python total and average
            python_avg_full = python_total_full / python_n_files if python_n_files else 0
            python_avg_after = python_total_after / python_n_files if python_n_files else 0
        message.append("")
            
    # Total and average
    avg_full = grand_total_full / grand_n_files if grand_n_files else 0
    avg_after = grand_total_after / grand_n_files if grand_n_files else 0

    message.append("-" * 60)
    message.append(f"{'JAVA_TOTAL':30} {java_total_full:10,} {java_total_after:12,}")
    message.append(f"{'JAVA_AVG':30} {java_avg_full:10.2f} {java_avg_after:12.2f}")
    message.append("")
    message.append("-" * 60)
    message.append(f"{'PYTHON_TOTAL':30} {python_total_full:10,} {python_total_after:12,}")
    message.append(f"{'PYTHON_AVG':30} {python_avg_full:10.2f} {python_avg_after:12.2f}")
    message.append("")
    message.append("-" * 60)
    message.append(f"{'GRAND_TOTAL':30} {grand_total_full:10,} {grand_total_after:12,}")
    message.append(f"{'GRAND_AVG':30} {avg_full:10.2f} {avg_after:12.2f}")
    message.append("")

    # Save
    with open(args.save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(message))

    print(f"✅  Results saved to '{args.save_path}'")


if __name__ == "__main__":
    main()