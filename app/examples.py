"""Preset inputs for the demo: GLUE validation examples with their gold labels.

For each task, two examples that WEMoE (``full-mean-k256-s0``) gets right and one it gets wrong.
``idx`` is the GLUE validation idx; ``texts`` follow the order of the task's text fields.
"""

EXAMPLES = {
    "mrpc": [
        {"idx": 9, "label": "equivalent", "wemoe_correct": True, "texts": (
            "He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .",
            "\" The foodservice pie business does not fit our long-term growth strategy .")},
        {"idx": 185, "label": "not_equivalent", "wemoe_correct": True, "texts": (
            "Friday , Stanford ( 47-15 ) blanked the Gamecocks 8-0 .",
            "Stanford ( 46-15 ) has a team full of such players this season .")},
        {"idx": 33, "label": "not_equivalent", "wemoe_correct": False, "texts": (
            "No dates have been set for the civil or the criminal trial .",
            "No dates have been set for the criminal or civil cases , but Shanley has pleaded not guilty .")},
    ],
    "rte": [
        {"idx": 43, "label": "entailment", "wemoe_correct": True, "texts": (
            "Pibul Songgram was the pro-Japanese military dictator of Thailand during World War 2.",
            "Pibul was the dictator of Thailand.")},
        {"idx": 25, "label": "not_entailment", "wemoe_correct": True, "texts": (
            "The San Diego Padres ace, Jake Peavy, was hurt in an 8-5 loss to the St. Louis Cardinals.",
            "The San Diego Padres won the game against the St. Louis Cardinals.")},
        {"idx": 81, "label": "entailment", "wemoe_correct": False, "texts": (
            "Hepburn's family will receive the proceeds from the sale.",
            "Proceeds go to Hepburn's family.")},
    ],
    "cola": [
        {"idx": 0, "label": "acceptable", "wemoe_correct": True, "texts": (
            "The sailors rode the breeze clear of the rocks.",)},
        {"idx": 4, "label": "unacceptable", "wemoe_correct": True, "texts": (
            "As you eat the most, you want the least.",)},
        {"idx": 6, "label": "unacceptable", "wemoe_correct": False, "texts": (
            "I demand that the more John eat, the more he pays.",)},
    ],
    "sst2": [
        {"idx": 0, "label": "positive", "wemoe_correct": True, "texts": (
            "it 's a charming and often affecting journey . ",)},
        {"idx": 4, "label": "negative", "wemoe_correct": True, "texts": (
            "it 's slow -- very , very slow . ",)},
        {"idx": 22, "label": "negative", "wemoe_correct": False, "texts": (
            "holden caulfield did it better . ",)},
    ],
    "qnli": [
        {"idx": 851, "label": "entailment", "wemoe_correct": True, "texts": (
            "What is the name of the professional skateboarder that lives in southern California?",
            "Professional skateboarder Tony Hawk, professional surfers Rob Machado, Tim Curran, Bobby Martinez, "
            "Pat O'Connell, Dane Reynolds, and Chris Ward, and professional snowboarder Shaun White live in "
            "southern California.")},
        {"idx": 7, "label": "not_entailment", "wemoe_correct": True, "texts": (
            "What is the name of the professional skateboarder that lives in southern California?",
            "Southern California is also important to the world of yachting.")},
        {"idx": 33, "label": "not_entailment", "wemoe_correct": False, "texts": (
            "A function problem is an example of what?",
            "Notable examples include the traveling salesman problem and the integer factorization problem.")},
    ],
}


def example_title(example: dict, task_name: str = "") -> str:
    """Dropdown label: the start of the first text, its gold label and whether WEMoE gets it right."""
    text = example["texts"][0].strip()
    short = text if len(text) <= 60 else text[:57].rstrip() + "..."
    verdict = "WEMoE right" if example["wemoe_correct"] else "WEMoE wrong"
    prefix = f"{task_name} " if task_name else ""
    return f"{prefix}#{example['idx']}: {short} [{example['label']}; {verdict}]"
