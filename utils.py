
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
- Do NOT include pronouns, articles, determiners, explanations, or any extra text.
- For multi-word expressions, inflect only the head word (main verb, noun, or adjective depending on the POS), and keep all other words unchanged and in place.
- Ensure correct spelling, accents, and irregular forms.
- Preserve the natural grammatical order of forms: base → derived forms; singular → plural; masculine → feminine.. For verbs, also follow the standard tense sequence for the language.
- If the term is not inflectable as required, return an empty list.

Examples:

Input: French verb \'parler\'
Requested forms: present indicative
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

Input: French verb \'parler\'
Requested forms: participe passé
Output: ['parlé', 'parlée', 'parlés', 'parlées']

Input: French verb \'grouper\'
Requested forms: subjonctif imparfait
Output: ['groupasse', 'groupasses', 'groupât', 'groupassions', 'groupassiez', 'groupassent']

Input: English noun \'box\'
Requested forms: number and possessive forms
Output: ['box', 'boxes', "box's", "boxes'"]

Input: English adj \'big\'
Requested forms: comparative/superlative forms
Output: ['big', 'bigger', 'biggest']

Input: Spanish adj \'bonito\'
Requested forms: formas de género y número
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

Input: English verb \'speak\'
Requested forms: base form, 3rd person singular present, simple past, past participle, present participle (-ing)
Output: ['speak', 'speaks', 'spoke', 'spoken', 'speaking']

Input: English verb \'to put on\'
Requested forms: base form, 3rd person singular present, simple past, past participle, present participle (-ing)
Output: ['put on', 'puts on', 'put on', 'put on', 'putting on']

Input: English acronym \'NASA\'
Requested forms: possessive forms
Output: ['NASA', 'NASAs', "NASA's", "NASAs'"]

Input: Spanish verb \'granizar\'
Requested forms: imperativo
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
    "proper noun": {
        "French": [
            "lowercase, UPPERCASE and Truecase",
        ],
        "Spanish": [
            "lowercase, UPPERCASE and Truecase",
        ],
        "English": [
            "possessive forms",
        ]
    },
    "acronym": {
        "French": [
            "lowercase, UPPERCASE and Truecase",
        ],
        "Spanish": [
            "lowercase, UPPERCASE and Truecase",
        ],
        "English": [
            "possessive forms",
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