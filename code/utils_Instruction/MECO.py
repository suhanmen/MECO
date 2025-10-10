import json
from typing import List, Dict
import pdb

SYSTEM_BASE_INTRO = (
    "You are the best programmer in the world.\n"
    "{expertise_guide}"
    "You will be asked to determine the time complexity of the following code.\n"
    "For the time complexity, choose one time complexity from the following options: "
    "'constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', and 'exponential'.\n"
    "Do not hesitate to use any other supplementary materials you need for the task.\n\n"
)
SIMPLE_JSON_GUIDE = (
    "Please output the time complexity of the entire code and the explanation for choosing that time complexity in JSON format.\n"
    "Json format should be:\n"
    "{\n"
    "  \"complexity\": \"time complexity of the whole code\",\n"
    "  \"explanation\": \"A selection description of the time complexity of the entire code\"\n"
    "}\n\n"
)
CONFIDENCE_JSON_GUIDE = (
    "Please output the time complexity of the entire code, along with a confidence score and an explanation for choosing that time complexity in JSON format.\n\n"
    "Confidence scores should be an integer from 1 to 10, where:\n"
    "  - 1 to 3: Very low confidence, highly uncertain.\n"
    "  - 4 to 6: Moderate confidence, but not completely certain.\n"
    "  - 7 to 8: High confidence, likely to be correct.\n"
    "  - 9 to 10: Very high confidence, almost certain.\n\n"
    "Json format should be:\n"
    "{\n"
    "  \"complexity\": \"<time complexity of the whole code>\",\n"
    "  \"confidence\": <integer from 1 to 10>,\n"
    "  \"explanation\": \"<A selection description of the time complexity of the entire code>\"\n"
    "}\n\n"
)
FEW_SHOT_HDR = (
    "I will first provide you with example codes and their corresponding time complexity labels.\n"
    "These examples include one sample for each of the 7 time complexity classes.\n\n"
)

EXPERTISE_GUIDE = {
    'constant': (
        "Constant time complexity means that the execution time of a function does not depend on the size of the input.\n"
        "Regardless of how large the input is, the function completes in a fixed number of operations.\n"
    ),
    'logn': (
        "Logarithmic complexity means that the number of operations grows proportionally to the logarithm of the input size.\n"
        "This often occurs in divide-and-conquer algorithms or binary search-like structures.\n\n"
        "## Logical Steps to Determine logarithmic time complexity:\n"
        "1. Identify if the input size is being reduced by a constant factor (e.g., half) at each iteration.\n"
        "2. Look for algorithms that involve binary search, tree traversal (balanced trees), or divide-and-conquer.\n"
        "3. Ensure the number of operations does not scale linearly but instead decreases exponentially.\n"
        "4. If the loop or recursion reduces the problem size logarithmically, classify it as the logarithmic complexity.\n"
    ),
    'linear': (
        "Linear complexity means that the execution time grows proportionally with the input size.\n"
        "This is typical in single-loop iterations over an array or list.\n"
    ),
    'nlogn': (
        "O(n log n) complexity is common in efficient sorting algorithms like Merge Sort or Quick Sort.\n"
        "It arises when a problem is divided into smaller subproblems while still iterating over the input.\n\n"
        "## Logical Steps to Determine nlogn time complexity:\n"
        "1. Identify if the input is being divided into smaller parts recursively (logarithmic factor).\n"
        "2. Ensure that a linear operation is performed at each level of recursion.\n"
        "3. Look for sorting algorithms like Merge Sort, Quick Sort (average case), or efficient divide-and-conquer solutions.\n"
        "4. If the algorithm involves dividing the problem and processing each part linearly, classify it as nlogn time complexity.\n"
    ),
    'quadratic': (
       "Quadratic complexity occurs when an algorithm has double nested loops, where each loop iteration depends on the input size.\n"
    ),
    'cubic': (
        "Cubic complexity occurs when an algorithm has three nested loops iterating over the input size.\n\n"
        "## Logical Steps to Determine cubic time complexity:\n"
        "1. Identify if there are three nested loops iterating from 0 to n.\n"
        "2. Ensure that each element is compared or processed against every pair of elements.\n"
        "3. Look for brute-force solutions that check all triplets in an input set.\n"
        "4. If the number of operations scales as the cube of the input size, classify it as cubic complexity.\n"
    ),
    'exponential': (
        "Exponential complexity occurs when the number of operations doubles with each additional input element.\n"
        "This is common in brute-force recursive algorithms, such as solving the Traveling Salesman Problem.\n\n"
        "## Logical Steps to Determine exponential time complexity:\n"
        "1. Identify if the function calls itself recursively, doubling the number of calls at each step.\n"
        "2. Look for recursion that does not significantly reduce the input size in each step.\n"
        "3. Check for exhaustive searches, backtracking algorithms, or recursive Fibonacci calculations.\n"
        "4. If the number of operations grows exponentially with input size, classify it as exponential complexity.\n"
    )
    
}

def get_expertise_guide(expertise):
    return f"You are also an expert in {expertise} time complexity.\n" + EXPERTISE_GUIDE[expertise] + "\n"

def not_expertise_guide(expertise):
    return f"You are not an expert in {expertise} time complexity.\n"

# few-shot example cache
_few_shot_cache: Dict[str, str] = {}
def get_few_shot_examples(language: str) -> str:
    if language not in _few_shot_cache:
        path = f"../data/7shot/{language}-7shot.jsonl"
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                examples.append(
                    f'{{"source code": "{obj["src"]}", "time complexity": "{obj["complexity"]}"}}'
                )
        _few_shot_cache[language] = "\n".join(examples) + "\n\n"
    return _few_shot_cache[language]





def build_single(src: str, **_) -> List[Dict[str, str]]:
    system = SYSTEM_BASE_INTRO.format(expertise_guide='') + SIMPLE_JSON_GUIDE
    user = (
        "----------------------------------------\n"
        f"{src}\n"
        "----------------------------------------\n"
        "Calculate the time complexity of the given code.\n"
        "Json format should be:\n"
        "{\n"
        '  "complexity": time complexity of the whole code,\n'
        '  "explanation": A selection description of the time complexity of the entire code\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def build_single_fewshot(src: str, language: str, **_) -> List[Dict[str, str]]:
    few = get_few_shot_examples(language)
    system = SYSTEM_BASE_INTRO.format(expertise_guide='') + FEW_SHOT_HDR + few + SIMPLE_JSON_GUIDE
    user = (
        "----------------------------------------\n"
        f"{src}\n"
        "----------------------------------------\n"
        "Calculate the time complexity of the given code.\n"
        "Json format should be:\n"
        "{\n"
        '  "complexity": time complexity of the whole code,\n'
        '  "explanation": A selection description of the time complexity of the entire code\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def build_single_codecomplex_expertise(src: str, expertise: str, **_) -> List[Dict[str, str]]:
    system = SYSTEM_BASE_INTRO.format(expertise_guide=get_expertise_guide(expertise)) + SIMPLE_JSON_GUIDE
    user = (
        "----------------------------------------\n"
        f"{src}\n"
        "----------------------------------------\n"
        "Calculate the time complexity of the given code.\n"
        "Json format should be:\n"
        "{\n"
        '  "complexity": time complexity of the whole code,\n'
        '  "explanation": A selection description of the time complexity of the entire code\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def build_single_codecomplex_expertise_with_confidence(src: str, expertise: str, **_) -> List[Dict[str, str]]:
    system = SYSTEM_BASE_INTRO.format(expertise_guide=get_expertise_guide(expertise)) + CONFIDENCE_JSON_GUIDE
    user = (
        "----------------------------------------\n"
        f"{src}\n"
        "----------------------------------------\n"
        "Calculate the time complexity of the given code.\n"
        "Json format should be:\n"
        "{\n"
        '  "complexity": <time complexity of the whole code>,\n'
        '  "confidence": <integer from 1 to 10>,\n'
        '  "explanation": <A selection description of the time complexity of the entire code>\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def _build_vote_based(src: str, expertise: str, vote_sentence: str, **_) -> List[Dict[str, str]]:
    system = (
        SYSTEM_BASE_INTRO.format(expertise_guide=get_expertise_guide(expertise)) +
        "After reading the code, you will be asked to compute its time complexity.\n"
        "Your response should be independent and well-reasoned based on the code.\n"
        "You may also refer to the following model predictions as reference.\n"
        "Output format:\n"
        "{complexity, explanation}\n\n"
    )
    user = (
        "----------------------------------------\n"
        f"Code to analyze:\n{src}\n"
        "----------------------------------------\n"
        "Predictions:\n\n"
        f"{vote_sentence}"
        "----------------------------------------\n"
        "Now compute the complexity and explain in JSON.\n"
        "{\n"
        '  "complexity": time complexity,\n'
        '  "explanation": rationale\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def _build_no_expertise_multi(src: str, vote_sentence: str, **_) -> List[Dict[str, str]]:
    system = (
        SYSTEM_BASE_INTRO.format(expertise_guide='') +
        "After reading the code, compute its time complexity independently.\n"
        "You may refer to these model predictions.\n"
        "Output in JSON {complexity, explanation}.\n\n"
    )
    user = (
        "----------------------------------------\n"
        f"Code to analyze:\n{src}\n"
        "----------------------------------------\n"
        f"{vote_sentence}"
        "----------------------------------------\n"
        "Compute complexity.\n"
        "{\n"
        '  "complexity": time complexity,\n'
        '  "explanation": rationale\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def _build_with_logit_no_expertise(src: str, vote_sentence: str, **_) -> List[Dict[str, str]]:
    system = (
        SYSTEM_BASE_INTRO.format(expertise_guide='') +
        "After reading the code, compute its time complexity.\n"
        "Refer also to predictions with confidence scores.\n"
        "Output in JSON {complexity, explanation}.\n\n"
    )
    user = (
        "----------------------------------------\n"
        f"Code to analyze:\n{src}\n"
        "----------------------------------------\n"
        f"{vote_sentence}"
        "----------------------------------------\n"
        "Compute complexity.\n"
        "{\n"
        '  "complexity": time complexity,\n'
        '  "explanation": rationale\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def _build_with_logit(src: str, expertise: str, vote_sentence: str, **_) -> List[Dict[str, str]]:
    system = (
        SYSTEM_BASE_INTRO.format(expertise_guide=get_expertise_guide(expertise)) +
        f"You are also an expert in {expertise} time complexity.\n\n"
        "After reading the code, compute its time complexity.\n"
        "Refer also to predictions with confidence scores.\n"
        "Output in JSON {complexity, explanation}.\n\n"
    )
    user = (
        "----------------------------------------\n"
        f"Code to analyze:\n{src}\n"
        "----------------------------------------\n"
        f"{vote_sentence}"
        "----------------------------------------\n"
        "Compute complexity.\n"
        "{\n"
        '  "complexity": time complexity,\n'
        '  "explanation": rationale\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def _build_judge_no_expertise(src: str, vote_sentence: str, **_) -> List[Dict[str, str]]:
    system = (
        "You are an expert judge. Choose the correct complexity based solely on these predictions.\n\n"
    )
    user = (
        f"{vote_sentence}\n"
        "Decide final complexity and explain.\n"
        "{\n"
        '  "complexity": time complexity,\n'
        '  "explanation": rationale\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def _build_judge_with_expertise(src: str, vote_sentence: str, **_) -> List[Dict[str, str]]:
    system = (
        "You are an expert judge with domain expertise. Choose the correct complexity based on these predictions and expertise.\n\n"
    )
    user = (
        f"{vote_sentence}\n"
        "Decide final complexity and explain.\n"
        "{\n"
        '  "complexity": time complexity,\n'
        '  "explanation": rationale\n'
        "}\n\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

PROMPT_BUILDERS: Dict[str, callable] = {
    # Single model setting
    'single': build_single,
    'single-fewshot': build_single_fewshot,
    
    # MECO setting
    'single-codecomplex-expertise': build_single_codecomplex_expertise,
    'multi-codecomplex-expertise': _build_vote_based,
    
    # MECO with confidence
    'single-codecomplex-expertise-with-confidence': build_single_codecomplex_expertise_with_confidence,
    
    # MECO multiple round setting
    'Round2': _build_vote_based,
    'Round3': _build_vote_based,
    'Round4': _build_vote_based,
    'Round5': _build_vote_based,
    
    # Ablation study
    'single-no-expertise_multi-no-expertise': _build_no_expertise_multi, # Based on single no expertise, multi no expertise
    'single-expertise_multi-no-expertise': _build_vote_based, # Based on single with expertise, multi no expertise
    'single-no-expertise_multi-expertise': _build_vote_based, # Based on single no expertise, multi with expertise
    
    # logit come from 'GENERATION_TYPE' -> model
    # This prompt use for debate mode
    'single-no-expertise_multi-no-expertise-with-logit': _build_with_logit_no_expertise, # Based on single no expertise, multi no expertise, with logit
    'single-no-expertise_multi-expertise-with-logit': _build_with_logit, # Based on single no expertise, multi with expertise, with logit
    'single-expertise_multi-no-expertise-with-logit': _build_with_logit_no_expertise, # Based on single with expertise, multi no expertise, with logit
    'multi-codecomplex-expertise-with-logit': _build_with_logit, # Both Single and Multi with expertise
    
    
    # Judge model setting
    'judge_model-Ns_Nm': _build_judge_no_expertise, # Ns means single no expertise, Nm means multi no expertise
    'judge_model-Ns_Ym': _build_judge_with_expertise, # Ns means single no expertise, Ym means multi with expertise
    'judge_model-Ys_Nm': _build_judge_with_expertise, # Ys means single with expertise, Nm means multi no expertise
    'judge_model-Ys_Ym': _build_judge_with_expertise, # Ys means single with expertise, Ym means multi with expertise
}

def get_prompt_messages(option: str, src: str, **kwargs) -> List[Dict[str, str]]:
    builder = PROMPT_BUILDERS.get(option)
    if builder is None:
        raise KeyError(f"No prompt builder defined for option '{option}'")
    return builder(src, **kwargs)
