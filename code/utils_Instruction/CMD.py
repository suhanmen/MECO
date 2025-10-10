# models 
#Agnet1, Agent2, Agent3 = "A", "B", "C"
#temperature = 0.25

from typing import List, Dict
import pdb

# template for initialize
INITIALIZE_SYSTEM_PROMPT = (
    "You will be asked to determine the time complexity of the following code.\n"
    "For the time complexity, choose one time complexity from the following options 'constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', and 'exponential'.\n"
    "Do not hesitate to use any other supplementary materials you need for the task.\n\n"
    "I will first give you the code.\n"
    "After you read the code,\n"
    "I will ask you to compute the time complexity of the code.\n\n"
    "Please output the time complexity of the entire code, and an explanation for choosing that time complexity in JSON format.\n\n"
)

ANSWER_FORMAT_SYSTEM = (
    "Json format should be:\n"
    "{\n"
    '  "complexity": "<time complexity of the whole code>",\n'
    '  "explanation": "<A selection description of the time complexity of the entire code>"\n'
    "}\n\n"
)

USER_PROMPT = (
    "Calculate the time complexity of the given code.\n"
    "Please output the time complexity of the whole code in a json format.\n"
    "Json format should be\n"
    "{\n"
    '    "complexity": time complexity of the whole code,\n'
    '    "explanation": A selection description of the time complexity of the entire code,\n'
    "}\n\n"
)

TIE_SYSTEM_PROMPT_GENERAL = (
    "Multiple groups of agents have proposed different time complexities for the same code, and their votes resulted in a tie.\n\n"
    "Each complexity class is supported by multiple agents with explanations.\n"
    "Your task is to analyze the different complexity claims and choose the one that is most plausible.\n"
    "Do not simply count votes. Base your decision on the reasoning quality.\n\n"
    "Please provide your final decision in the following JSON format:\n"
    "{\n"
    '  "final_decision": "<selected complexity>",\n'
    '  "justification": "<your reasoning why this complexity is most plausible>"\n'
    "}\n"
)

discussion_system_prompt = (
    "There are 2 groups of people discussing on the same topic. I will provide you the detailed opinions and reasoning steps from your group member and opinions from other group members.\n"
    "Use these opinions and your previous opinion as additional advice, note that they maybe wrong.\n"
    "Do not copy other’s entire answer, modify the part you believe is wrong.\n"
    
)


def build_initialize(src: str, **_) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": INITIALIZE_SYSTEM_PROMPT + ANSWER_FORMAT_SYSTEM},
        {"role": "user", "content": f"----------------------------------------\n{src}\n----------------------------------------\n" + USER_PROMPT}
    ]


def build_discussion(
    src: str,
    in_group_data: List[Dict],   # same group agent's output
    out_group_data: List[Dict],  # different group agent's output
    **_
) -> List[Dict[str, str]]:

    # same group information
    in_group_lines = ["Your group’s opinions:"]
    for i, agent in enumerate(in_group_data):
        in_group_lines.append(
            f"- Agent {i+1}: Answer: {agent['complexity']}, Reason: {agent['explanation']}"
        )

    # different group information
    out_group_lines = ["Other group members’ opinions:"]
    for i, agent in enumerate(out_group_data):
        out_group_lines.append(
            f"- Agent {i+1+len(in_group_data)}: Answer: {agent['complexity']}"
        )

    system_prompt = (
        INITIALIZE_SYSTEM_PROMPT
        + "\n"
        + discussion_system_prompt
        + "\n"
        + "\n".join(in_group_lines)
        + "\n\n"
        + "\n".join(out_group_lines)
        + "\n\n"
        + ANSWER_FORMAT_SYSTEM
    )

    user_prompt = (
        f"----------------------------------------\n{src}\n----------------------------------------\n"
        + USER_PROMPT
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def build_tie(src: str, candidates: List[Dict[str, object]], **_) -> List[Dict[str, str]]:
    complexity_section = []
    for idx, cand in enumerate(candidates):
        header = f"{len(cand['explanations'])} agents:\n- Complexity: {cand['complexity']} ({len(cand['explanations'])} supporting agents)"
        explanation_lines = [f"- Explanation: {exp}" for exp in cand.get("explanations", [])]
        section = header + "\n" + "\n".join(explanation_lines)
        complexity_section.append(section)

    user_msg = (
        f"Here is the code to evaluate:\n"
        f"----------------------------------------\n{src}\n----------------------------------------\n\n"
        "The agents were divided in their opinions, resulting in a tie among the following complexity classes:\n\n" +
        "\n\n".join(complexity_section) +
        "\n\nPlease select the most plausible complexity class among the above options."
    )

    return [
        {"role": "system", "content": TIE_SYSTEM_PROMPT_GENERAL},
        {"role": "user", "content": user_msg}
    ]




PROMPT_BUILDERS: Dict[str, callable] = {
    "CMD-Group1": build_initialize,
    "CMD-Group2": build_initialize,
    "CMD-Group1_discussion1": build_discussion,
    "CMD-Group2_discussion1": build_discussion,
    "CMD-Group1_discussion2": build_discussion,
    "CMD-Group2_discussion2": build_discussion,
    "CMD-tie": build_tie,
}

def get_prompt_messages(option: str, src: str, **kwargs) -> List[Dict[str, str]]:
    builder = PROMPT_BUILDERS.get(option)
    if not builder:
        raise KeyError(f"No prompt builder for option '{option}'")
    return builder(src, **kwargs)