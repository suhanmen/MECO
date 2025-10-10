from typing import List, Dict

AFFIRMATIVE_SYSTEM_PROMPT = (
    "You are affirmative side. Please express your viewpoints.\n"
)

NEGATIVE_SYSTEM_PROMPT = (
    "You are negative side. You disagree with the affirmative side’s points. Provide your reasons and answer.\n"
)

JUDGE_DISCUSSION_PROMPT = (
    "You are the best programmer in the world.\n"
    "You are a debater. Hello and welcome to the debate competition.\n"
    "It’s not necessary to fully agree with each other’s perspectives, as our objective is to find the correct answer.\n"
    "You will be asked to determine the time complexity of the following code.\n"
    "For the time complexity, choose one from the following: 'constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential'.\n"
    "Do not hesitate to use any other supplementary materials.\n\n"
    "I will first give you the code. After reading it, compute the time complexity.\n"
)


JUDGE_MODEL_PROMPT = (
    "You are a moderator.\n"
    "There will be two debaters involved in a debate competition.\n"
    "They will present their answers and discuss their perspectives on the problem of determining the time complexity of the following code.\n"
    "At the end of each round, you will evaluate both sides’ answers and decide which one is correct.\n"
)
##########################################################
ANSWER_FORMAT = (
    "Json format should be:\n"
    "{\n"
    "  \"complexity\": \"<time complexity of the whole code>\",\n"
    "  \"explanation\": \"<description of why this complexity was chosen>\"\n"
    "}\n\n"
)

JUDGE_ANSWER_FORMAT = (
    "Json format should be:\n"
    "{\n"
    '  "complexity": "<chosen time complexity or \'unknown\'>",\n'
    '  "explanation": "<reason for the judgment>",\n'
    '  "pass": 1 or 0  // 1 means the debate is resolved, 0 means continue debating\n'
    "}\n\n"
)

USER_PROMPT = (
    "Calculate the time complexity of the given code.\n"
    "Please output the result in JSON format as described.\n"
    "----------------------------------------\n"
    "{code}\n"
    "----------------------------------------\n"
)

def build_affirmative(src: str, previous_negative_output: str = "", **_) -> List[Dict[str, str]]:
    context = AFFIRMATIVE_SYSTEM_PROMPT

    if previous_negative_output and previous_negative_output.strip().lower() not in ["", "none", "null"]:
        context += (
            "\nHere is what the negative side has said:\n\n" +
            previous_negative_output +
            "\n\nNow, provide your counter-argument."
        )

    return [
        {"role": "system", "content": context + JUDGE_DISCUSSION_PROMPT + ANSWER_FORMAT},
        {"role": "user", "content": USER_PROMPT.format(code=src)}
    ]


def build_negative(src: str, previous_affirmative_output: str = "", **_) -> List[Dict[str, str]]:
    context = NEGATIVE_SYSTEM_PROMPT
    if previous_affirmative_output:
        context += (
            "\nHere is what the affirmative side has said:\n\n" +
            previous_affirmative_output +
            "\n\nNow, provide your rebuttal."
        )

    return [
        {"role": "system", "content": context + JUDGE_DISCUSSION_PROMPT + ANSWER_FORMAT},
        {"role": "user", "content": USER_PROMPT.format(code=src)}
    ]


def build_judge(src: str,affirmative_vote_sentence: str = "",negative_vote_sentence: str = "",**_) -> List[Dict[str, str]]:

    system_msg = (
        "You are the judge.\n"
        "Based on the voting results from both the affirmative and negative sides, decide which one is more convincing.\n\n"
        "Affirmative results:\n" + affirmative_vote_sentence +
        "\n\nNegative results:\n" + negative_vote_sentence +
        "\n\n" + JUDGE_ANSWER_FORMAT
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": USER_PROMPT.format(code=src)}
    ]





PROMPT_BUILDERS: Dict[str, callable] = {
    "MAD-affirmative_discussion1": build_affirmative,
    "MAD-negative_discussion1": build_negative,
    "MAD-affirmative_discussion2": build_affirmative,
    "MAD-negative_discussion2": build_negative,
    "MAD-affirmative_discussion3": build_affirmative,
    "MAD-negative_discussion3": build_negative,
    "MAD-affirmative_discussion4": build_affirmative,
    "MAD-negative_discussion4": build_negative,
    "MAD-affirmative_discussion5": build_affirmative,
    "MAD-negative_discussion5": build_negative,
    "MAD-judge1": build_judge,
    "MAD-judge2": build_judge,
    "MAD-judge3": build_judge,
    "MAD-judge4": build_judge,
    "MAD-judge5": build_judge,
}

def get_prompt_messages(option: str, src: str, **kwargs) -> List[Dict[str, str]]:
    builder = PROMPT_BUILDERS.get(option)
    if not builder:
        raise KeyError(f"No prompt builder for option '{option}'")
    return builder(src, **kwargs)
