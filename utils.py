
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
- Generate forms based on the part of speech (POS) and language-specific rules.
- Do NOT include pronouns, articles, determiners, explanations, or any extra text.
- For multi-word expressions, inflect only the head word (main verb, noun, or adjective depending on the POS) and keep all other words unchanged and in place.
- For multi-word expressions (e.g., "put on", "take off"), always:
  1. Identify the head word (first word),
  2. Inflect the head word,
  3. Append the particle/s unchanged to each form.  
- Preserve the natural grammatical order of forms (e.g., base → derived forms; singular → plural; masculine → feminine). For verbs, follow the standard tense sequence for the language.
- Only provide valid inflections that exist in the language, ensuring correct spelling, accents, and irregular forms.
- If the term is not inflectable as required, return an empty list.

Examples:

Input: French verb 'parler'
Requested forms: present indicative
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

Input: French verb 'parler'
Requested forms: participe passé
Output: ['parlé', 'parlée', 'parlés', 'parlées']

Input: French verb 'grouper'
Requested forms: subjonctif imparfait
Output: ['groupasse', 'groupasses', 'groupât', 'groupassions', 'groupassiez', 'groupassent']

Input: English noun 'box'
Requested forms: number and possessive forms
Output: ['box', 'boxes', "box's", "boxes'"]

Input: English adj 'big'
Requested forms: comparative/superlative forms
Output: ['big', 'bigger', 'biggest']

Input: Spanish adj 'bonito'
Requested forms: formas de género y número
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

Input: English verb 'speak'
Requested forms: base form, 3rd person singular present, simple past, past participle, present participle (-ing)
Output: ['speak', 'speaks', 'spoke', 'spoken', 'speaking']

Input: English verb 'to put on'
Requested forms: base form, 3rd person singular present, simple past, past participle, present participle (-ing)
Output: ['put on', 'puts on', 'put on', 'put on', 'putting on']

Input: English acronym 'NASA'
Requested forms: possessive forms
Output: ['NASA', 'NASAs', "NASA's", "NASAs'"]

Input: Spanish verb 'granizar'
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

def fix_term(curr_term):
    curr_term = curr_term.lower()
    lemma = None
    pos = None
    if curr_term.find("(") != -1 and curr_term.find(")") != -1:
        begin = curr_term.find("(")
        end = curr_term.find(")")        
        pos = curr_term[begin+1:end]
        if pos.startswith("proper noun"):
            pos = "(proper noun)"
        lemma = curr_term[:begin].strip()
        if lemma.startswith("to ") and pos == "verb":
            lemma = lemma[3:] # remove "to " from verb infinitive form in English (e.g. "to speak" -> "speak")
        if lemma.find(" ") != -1 and pos in ["verb", "noun", "adj"]:
            pos = pos + ' phrase'
        curr_term = f"{lemma} ({pos})"
    return curr_term, lemma, pos

