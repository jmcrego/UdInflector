
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

#- Do NOT include pronouns, articles, determiners, explanations, or any extra text.
#- For multi-word expressions, inflect only the head word and keep other components unchanged.


PROMPT_PREFIX = """You are a professional linguist specializing in term inflection (including verb conjugation).

Task:
- Output ONLY a Python list with the requested conjugated/inflected forms of the given term and language.
- Use the translation field to disambiguate the meaning of the term to conjugate/inflect.
- Preserve a consistent and linguistically standard ordering of forms.
- For multi-word expressions, inflect the whole expression.
- Use standard modern orthography conventions for each language.
- Only provide valid inflections that exist in the language (irregular forms included).
- Do not provide synonyms or related words, only inflected forms of the given term.
- If the term is not inflectable as requested, return an empty list.

Examples:

INFLECT(language='French', term='parler', translation='to speak' , request='present indicative')
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

INFLECT(language='French', term='parler', translation='to speak', request='participe passé')
Output: ['parlé', 'parlée', 'parlés', 'parlées']

INFLECT(language='French', term='grouper', translation='to group', request='subjonctif imparfait')
Output: ['groupasse', 'groupasses', 'groupât', 'groupassions', 'groupassiez', 'groupassent']

INFLECT(language='English', term='box', translation='caja', request='number and possessive forms')
Output: ['box', 'boxes', "box's", "boxes'"]

INFLECT(language='English', term='big', translation='grande', request='comparative/superlative forms')
Output: ['big', 'bigger', 'biggest']

INFLECT(language='Spanish', term='bonito', translation='beau', request='formas de género y número')
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

INFLECT(language='English', term='speak', translation='hablar', request='base form')
Output: ['speak']

INFLECT(language='English', term='speak', translation='hablar', request='present participle (-ing)')
Output: ['speaking']

INFLECT(language='English', term='go back', translation='volver', request='simple past')
Output: ['went back']

INFLECT(language='English', term='NASA', translation='NASA', request='possessive forms')
Output: ['NASA', 'NASAs', "NASA's", "NASAs'"]

INFLECT(language='Spanish', term='granizar', translation='grêler', request='formas de género y número')
Output: []

"""

PROMPT_PREFIX_old = """You are a professional linguist specializing in term inflection (including verb conjugation).

Task:
- Output ONLY a Python list with the requested conjugated/inflected forms of the given term and language.
- Preserve a consistent and linguistically standard ordering of forms.
- For multi-word expressions, inflect the whole expression.
- Use standard modern orthography conventions for each language.
- Only provide valid inflections that exist in the language (irregular forms included).
- Do not provide synonyms or related words, only inflected forms of the given term.
- If the term is not inflectable as requested, return an empty list.

Examples:

INFLECT(language='French', pos='verb', term='parler', request='present indicative')
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

INFLECT(language='French', pos='verb', term='parler', request='participe passé')
Output: ['parlé', 'parlée', 'parlés', 'parlées']

INFLECT(language='French', pos='verb', term='grouper', request='subjonctif imparfait')
Output: ['groupasse', 'groupasses', 'groupât', 'groupassions', 'groupassiez', 'groupassent']

INFLECT(language='English', pos='noun', term='box', request='number and possessive forms')
Output: ['box', 'boxes', "box's", "boxes'"]

INFLECT(language='English', pos='adj', term='big', request='comparative/superlative forms')
Output: ['big', 'bigger', 'biggest']

INFLECT(language='Spanish', pos='adj', term='bonito', request='formas de género y número')
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

INFLECT(language='English', pos='verb', term='speak', request='base form')
Output: ['speak']

INFLECT(language='English', pos='verb', term='speak', request='present participle (-ing)')
Output: ['speaking']

INFLECT(language='English', pos='verb', term='go back', request='simple past')
Output: ['went back']

INFLECT(language='English', pos='acronym', term='NASA', request='possessive forms')
Output: ['NASA', 'NASAs', "NASA's", "NASAs'"]

INFLECT(language='Spanish', pos='verb', term='granizar', request='formas de género y número')
Output: []

"""


REQUESTS = {
    "verb": {
        "French": [
            "infinitif",
            "participe présent",
            "participe passé",
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
            "infinitivo",
            "gerundio", 
            "participio",
            "presente indicativo",
            "pretérito indefinido",
            "imperativo",
            "futuro simple",
            "subjuntivo presente",
            "condicional simple",
            "subjuntivo imperfecto"
        ],
        "English": [
            "base form", 
            "3rd person singular present", 
            "simple past", 
            "past participle", 
            "present participle (-ing)"
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

REQUESTS_old = {
    "verb": {
        "French": [
            "infinitif",
            "participe présent",
            "participe passé",
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
            "infinitivo",
            "gerundio", 
            "participio",
            "presente indicativo",
            "pretérito indefinido",
            "imperativo",
            "futuro simple",
            "subjuntivo presente",
            "condicional simple",
            "subjuntivo imperfecto"
        ],
        "English": [
            "base form", 
            "3rd person singular present", 
            "simple past", 
            "past participle", 
            "present participle (-ing)"
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

