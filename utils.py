
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
- Output ONLY a Python list containing the correctly inflected forms of the given term.
- The "request" parameter ALWAYS specifies:
  1) the part of speech (POS), and
  2) the exact inflectional forms to generate.
- You MUST strictly follow the requested POS and forms.

Rules:
- Use the provided translation ONLY to disambiguate the meaning of the term.
- If the term does not match the requested POS, or if the requested inflection is not applicable to the term, return an empty list.

Inflection constraints:
- Only include valid, attested inflected forms (including irregular ones).
- Do NOT include synonyms, derivations, or related words.
- For multi-word expressions, inflect the full expression appropriately.
- Use standard modern orthography for the language.

Output format:
- Return ONLY a Python list of strings.
- No explanations, comments, or additional text.

Examples:

INFLECT(language='French', term='parler', translation='speak' , request='verb - présent indicatif (1s, 2s, 3s, 1p, 2p, 3p)')
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

INFLECT(language='French', term='parler', translation='speak', request='verb - participe passé (masc sg, fem sg, masc pl, fem pl)')
Output: ['parlé', 'parlée', 'parlés', 'parlées']

INFLECT(language='French', term='grouper', translation='group', request='verb - subjonctif imparfait (1s, 2s, 3s, 1p, 2p, 3p)')
Output: ['groupasse', 'groupasses', 'groupât', 'groupassions', 'groupassiez', 'groupassent']

INFLECT(language='English', term='box', translation='caja', request='noun - singular, plural, singular possessive, plural possessive')
Output: ['box', 'boxes', "box's", "boxes'"]

INFLECT(language='English', term='big', translation='grande', request='adj - comparative/superlative forms')
Output: ['big', 'bigger', 'biggest']

INFLECT(language='Spanish', term='bonito', translation='beau', request='adj - masc sg, fem sg, masc pl, fem pl')
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

INFLECT(language='English', term='speak', translation='hablar', request='verb - base form')
Output: ['speak']

INFLECT(language='English', term='speak', translation='hablar', request='verb - present participle (-ing)')
Output: ['speaking']

INFLECT(language='English', term='go back', translation='volver', request='verb - simple past')
Output: ['went back']

INFLECT(language='English', term='NASA', translation='NASA', request='noun - singular, plural, singular possessive, plural possessive')
Output: ['NASA', 'NASAs', "NASA's", "NASAs'"]

INFLECT(language='Spanish', term='granizar', translation='grêler', request='noun - masc sg, fem sg, masc pl, fem pl')
Output: []

INFLECT(language='English', term='bear', translation='llevar', request='verb - simple past')
Output: ['bore']

INFLECT(language='English', term='bear', translation='oso', request='noun - masc sg, fem sg, masc pl, fem pl')
Output: ['bear', 'bear', 'bears', 'bears']

INFLECT(language='English', term='bear', translation='oso', request='verb - past participle')
Output: []

"""


REQUESTS = {
    "French": [
        "verb - infinitif",
        "verb - participe présent",
        "verb - participe passé (masc sg, fem sg, masc pl, fem pl)",
        "verb - présent indicatif (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - imparfait indicatif (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - impératif (2s, 1p, 2p)",
        "verb - futur simple (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - subjonctif présent (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - conditionnel présent (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - passé simple (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - subjonctif imparfait (1s, 2s, 3s, 1p, 2p, 3p)",
        "noun - masc sg, fem sg, masc pl, fem pl",
        "adj - masc sg, fem sg, masc pl, fem pl",
    ],
    "Spanish": [
        "verb - infinitivo",
        "verb - gerundio", 
        "verb - participio (masc sg, fem sg, masc pl, fem pl)",
        "verb - presente indicativo (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - pretérito indefinido (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - imperativo afirmativo (2s, 3s, 1p, 2p, 3p)",
        "verb - futuro simple (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - subjuntivo presente (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - condicional simple (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - subjuntivo imperfecto (-se) (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - subjuntivo imperfecto (-ra) (1s, 2s, 3s, 1p, 2p, 3p)",
        "noun - masc sg, fem sg, masc pl, fem pl",
        "adj - masc sg, fem sg, masc pl, fem pl",
    ],
    "English": [
        "verb - base form", 
        "verb - 3rd person singular present", 
        "verb - simple past", 
        "verb - past participle", 
        "verb - present participle (-ing)",
        "noun - singular, plural, singular possessive, plural possessive",
        "adj - comparative/superlative forms",
    ],
}


