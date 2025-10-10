# Instruction/RECONCILE.py

import json
import pdb
from typing import List, Dict
from collections import Counter

# Agent identifiers and default temperature (current selection options)
#AGENT1, AGENT2, AGENT3 = "A", "B", "C"
#temperature = 0.7

# Define system and user fragments
INITIALIZE_PROMPT = (
    "You are the best programmer in the world.\n"
    "You will be asked to determine the time complexity of the following code.\n"
    "For the time complexity, choose one from 'constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential'.\n"
    "Do not hesitate to use any supplementary materials.\n\n"
    "I will first give you the code. After reading it, compute its time complexity.\n\n"
    "Also, evaluate your confidence level (0.0–1.0) and provide step-by-step reasoning.\n\n"
)

USER_FORMAT = (
    "Calculate the time complexity of the given code.\n"
    "Please output the time complexity of the whole code in JSON format.\n"
    "Json format should be:\n"
    "{\n"
    '    "complexity": time complexity of the whole code,\n'
    '    "confidence": <float between 0.0 and 1.0>,\n'
    '    "explanation": A selection description of the time complexity of the entire code\n'
    "}\n\n"
)

ANSWER_FORMAT_SYSTEM = (
    "Json format should be:\n"
    "{\n"
    '  "complexity": "<time complexity of the whole code>",\n'
    '  "confidence": <float between 0.0 and 1.0>,\n'
    '  "explanation": "<A selection description of the time complexity of the entire code>"\n'
    "}\n\n"
)



def build_initialize(src: str, **_) -> List[Dict[str, str]]:
    system = INITIALIZE_PROMPT + ANSWER_FORMAT_SYSTEM
    user = (
        "----------------------------------------\n"
        f"{src}\n"
        "----------------------------------------\n"
        + USER_FORMAT
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

from collections import Counter
from typing import List, Dict
import json

def build_discussion(src: str, vote_sentence: str, **_) -> List[Dict[str, str]]:
    try:
        agent_outputs = json.loads(vote_sentence)
    except json.JSONDecodeError:
        agent_outputs = []

    if not agent_outputs:
        system = INITIALIZE_PROMPT + "\n\nNo agent output found.\n\n" + ANSWER_FORMAT_SYSTEM
        user = (
            "----------------------------------------\n"
            f"{src}\n"
            "----------------------------------------\n"
            + USER_FORMAT
        )
        return [{"role": "system", "content": system},
                {"role": "user",   "content": user}]

    # 1. Calculate the number of votes
    vote_counter = Counter([a["complexity"] for a in agent_outputs])
    class_counts = vote_counter.most_common()
    max_count = class_counts[0][1]
    majority_classes = [cls for cls, count in class_counts if count == max_count]

    section = ""
    instruction = ""

    if max_count == 3:
        # 🟢 Case 1: 3:0 (All same)
        majority_group = agent_outputs
        majority_cls = majority_classes[0]
        section += f"There are {len(majority_group)} agents who think the answer is '{majority_cls}':\n"
        for a in majority_group:
            section += f"- Reason: {a['explanation']} Answer: {a['complexity']}, Confidence: {a['confidence']}\n"
        instruction = "Since all agents agree, carefully verify the consensus and explain whether you agree or not."

    elif max_count == 2:
        # 🟡 Case 2: 2:1 (majority vs minority)
        majority_cls = majority_classes[0]
        majority_group = [a for a in agent_outputs if a["complexity"] == majority_cls]
        minority_group = [a for a in agent_outputs if a["complexity"] != majority_cls]

        section += f"There are {len(majority_group)} agents who think the answer is '{majority_cls}':\n"
        for a in majority_group:
            section += f"- Reason: {a['explanation']} Answer: {a['complexity']}, Confidence: {a['confidence']}\n"

        minority_classes = sorted(set([a["complexity"] for a in minority_group]))
        minority_class_str = ", ".join(minority_classes)

        section += f"\nThere is {len(minority_group)} agent who thinks the answer is '{minority_class_str}':\n"
        for a in minority_group:
            section += f"- Reason: {a['explanation']} Answer: {a['complexity']}, Confidence: {a['confidence']}\n"

        instruction = "Clearly state which group you agree with and provide reasoning."

    else:
        # 🔴 Case 3: 1:1:1 (All different)
        majority_group = agent_outputs
        majority_classes_all = sorted(set([a["complexity"] for a in agent_outputs]))
        class_list_str = ", ".join(majority_classes_all)

        section += f"There are {len(agent_outputs)} agents who think the answer is one of [{class_list_str}]:\n"
        for a in agent_outputs:
            section += f"- Reason: {a['explanation']} Answer: {a['complexity']}, Confidence: {a['confidence']}\n"

        instruction = "No majority exists. Evaluate each opinion and choose the one you support, with justification."

    # Final prompt construction
    system = (
        INITIALIZE_PROMPT
        + "\n\n"
        + instruction
        + "\n\n"
        + section
        + "\n"
        + ANSWER_FORMAT_SYSTEM
    )

    user = (
        "----------------------------------------\n"
        f"{src}\n"
        "----------------------------------------\n"
        + USER_FORMAT
    )

    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]






PROMPT_BUILDERS: Dict[str, callable] = {
    "REC-initialize": build_initialize,
    "REC-discussion1":  build_discussion,
    "REC-discussion2":  build_discussion,
}

def get_prompt_messages(option: str, src: str, **kwargs) -> List[Dict[str, str]]:
    builder = PROMPT_BUILDERS.get(option)
    if not builder:
        raise KeyError(f"No prompt builder for option '{option}'")
    return builder(src, **kwargs)