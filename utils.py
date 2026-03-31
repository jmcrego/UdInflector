
def get_dtype_for_gpu():
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Running on GPU: {gpu_name}")
        if "V100" in gpu_name:
            return 'float16'
        elif "A100" in gpu_name or "H100" in gpu_name:
            return 'bfloat16'
        else:
            print("Unknown GPU type, defaulting to float16")
            return 'float16'
    print("No GPU detected, defaulting to float16")
    return 'float16'


PROMPT_PREFIX = """You are a professional linguist specializing in term inflection (including verb conjugation).

Task:
- Output ONLY a Python list with the required conjugated/inflected forms of the term.
- Guide your conjugation/inflection based on the part of speech (POS) and language-specific rules for the given term.
- Only provide inflections that exist in the given language.
- Do NOT include pronouns, explanations, or extra text.
- Ensure correct spelling, accents, and irregular forms.
- Preserve the natural grammatical order of forms: base → derived forms; masculine → feminine; singular → plural. For verbs, also follow the standard tense sequence for the language.
- If the term is not inflectable as required, return an empty list.

Examples:

Input: French verb \'parler\'
Forms: present indicative
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

Input: French verb \'parler\'
Forms: participe passé
Output: ['parlé', 'parlée', 'parlés', 'parlées']

Input: English noun \'box\'
Forms: number and possessive forms
Output: ['box', 'boxes', "box's", "boxes'"]

Input: English adj \'big\'
Forms: comparative/superlative forms
Output: ['big', 'bigger', 'biggest']

Input: Spanish adj \'bonito\'
Forms: formas de género y número
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

Input: English verb \'speak\'
Forms: base form, 3rd person singular present, simple past, past participle, present participle (-ing)
Output: ['speak', 'speaks', 'spoke', 'spoken', 'speaking']

Input: Spanish verb \'granizar\'
Forms: imperativo
Output: []

"""


INFLECTIONS = {
    "verb": {
        "French": [
            "infinitif, participe présent, participe passé",
            "présent indicatif",
            "imparfait indicatif",
            "impératif",
            "futur simple",
            "subjonctif présent",
            "conditionnel présent",
            "passé simple",
            "subjonctif imparfait"
        ],
        "Spanish": [
            "infinitivo, gerundio, participio",
            "presente indicativo",
            "pretérito indefinido",
            "imperativo",
            "futuro simple",
            "subjuntivo presente",
            "condicional simple",
            "subjuntivo imperfecto"
        ],
        "English": [
            "base form, 3rd person singular present, simple past, past participle, present participle (-ing)"
        ]
    },
    "noun": {
        "French": [
            "formes de genre et de nombre",
        ],
        "Spanish": [
            "formas de género y número",
        ],
        "English": [
            "number and possessive forms",
        ]
    },
    "adj": {
        "French": [
            "formes de genre et de nombre",
        ],
        "Spanish": [
            "formas de género y número",
        ],
        "English": [
            "comparative/superlative forms",
        ]
    }        
}