def load_MARKER(path):
    if 'Qwen2.5-Coder' in path:
        MARKER = "<|im_start|>assistant\n"
    elif 'Mistral-7B-Instruct' in path:
        MARKER = "[/INST] "
    elif 'Ministral-8B-Instruct' in path:
        MARKER = "[/INST]"
    elif 'deepseek-coder' in path:
        MARKER = "Response:\n"
    elif 'Meta-Llama-3.1' in path:
        MARKER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif 'codegemma-7b-it' in path:
        MARKER = "<start_of_turn>model\n"
    elif 'CodeLlama-7b-Instruct' in path:
        MARKER = "[/INST]  "
    elif "DeepSeek-R1-Distill" in path:
        MARKER = "<｜Assistant｜>"
    else:
        print("no marker")
        MARKER=""
    return MARKER

def label_format(label):
    """
    Convert integer label to complexity string.
    """
    if label == 0:
        return "constant"
    elif label == 1:
        return "logn"
    elif label == 2:
        return "linear"
    elif label == 3:
        return "nlogn"
    elif label == 4:
        return "quadratic"
    elif label == 5:
        return "cubic"
    elif label == 6:
        return "exponential"
    else:
        return "error"


def int_label_format(label):
    """
    Convert complexity string to integer label.
    """
    if label == "constant":
        return 0
    elif label == "logn":
        return 1
    elif label == "linear":
        return 2
    elif label == "nlogn":
        return 3
    elif label == "quadratic":
        return 4
    elif label == "cubic":
        return 5
    elif label == "exponential":
        return 6
    else:
        return -1


def model_name_format(model_name):
    """
    Map shorthand to full HF model ID.
    """
    mapping = {
        "deepseek": "deepseek-coder-7b-instruct-v1.5",
        "llama": "Meta-Llama-3.1-8B-Instruct",
        "qwen": "Qwen2.5-Coder-7B-Instruct",
        "ministral": "Ministral-8B-Instruct-2410",
        "mistral": "Mistral-7B-Instruct-v0.3",
        "codegemma": "codegemma-7b-it",
        "codellama": "CodeLlama-7b-Instruct-hf",
        "qwen-llm": "Qwen2.5-Coder-32B-Instruct",
        "llama3-llm": "Llama-3.3-70B-Instruct",
        "llama1-llm": "Llama-3.1-70B-Instruct",
        "deepseek-llm": "deepseek-coder-33b-instruct",
        "deepseekr1-qwen": "DeepSeek-R1-Distill-Qwen-7B",
        "deepseekr1-llama": "DeepSeek-R1-Distill-Llama-8B",
        
    }
    return mapping.get(model_name, "error")


def back_name_format(full_name):
    """
    Extract shorthand key from full model name.
    """
    reverse = {
        "deepseek-coder-7b-instruct-v1.5": "deepseek",
        "Meta-Llama-3.1-8B-Instruct": "llama",
        "Qwen2.5-Coder-7B-Instruct": "qwen",
        "Ministral-8B-Instruct-2410": "ministral",
        "DeepSeek-R1-Distill-Qwen-7B": "deepseekr1-qwen",
        "DeepSeek-R1-Distill-Llama-8B": "deepseekr1-llama",
    }
    for key, short in reverse.items():
        if full_name.startswith(key):
            return short
    return "error"


def get_best_expertise(expertise_map):
    """
    Return the key with the highest value in expertise_map.
    """
    if not expertise_map:
        return None
    try:
        return max(expertise_map, key=expertise_map.get)
    except Exception:
        pdb.set_trace()
        return None
