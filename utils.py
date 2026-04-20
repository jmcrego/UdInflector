
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


PROMPT_PREFIX2 = """You are a professional linguist specializing in term inflection (including verb conjugation).

Task:
- Output ONLY a Python list with the requested conjugated/inflected forms of the given term and language.
- Use the given translation to disambiguate the term meaning.
- Preserve a consistent and linguistically standard ordering of forms.
- For multi-word expressions, inflect the whole expression.
- Use standard modern orthography conventions for each language.
- Only provide valid inflections that exist in the language (irregular forms included).
- Do not provide synonyms or related words, only inflected forms of the given term.
- If the term is not inflectable as requested, return an empty list.

Examples:

INFLECT(language='French', term='parler', translation='to speak' , request='verb - present indicative')
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

INFLECT(language='French', term='parler', translation='to speak', request='verb - participe passé')
Output: ['parlé', 'parlée', 'parlés', 'parlées']

INFLECT(language='French', term='grouper', translation='to group', request='verb - subjonctif imparfait')
Output: ['groupasse', 'groupasses', 'groupât', 'groupassions', 'groupassiez', 'groupassent']

INFLECT(language='English', term='box', translation='caja', request='noun - number and possessive forms')
Output: ['box', 'boxes', "box's", "boxes'"]

INFLECT(language='English', term='big', translation='grande', request='adj - comparative/superlative forms')
Output: ['big', 'bigger', 'biggest']

INFLECT(language='Spanish', term='bonito', translation='beau', request='adj - formas de género y número')
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

INFLECT(language='English', term='speak', translation='hablar', request='verb - base form')
Output: ['speak']

INFLECT(language='English', term='speak', translation='hablar', request='verb - present participle (-ing)')
Output: ['speaking']

INFLECT(language='English', term='go back', translation='volver', request='verb - simple past')
Output: ['went back']

INFLECT(language='English', term='NASA', translation='NASA', request='noun - possessive forms')
Output: ['NASA', 'NASAs', "NASA's", "NASAs'"]

INFLECT(language='Spanish', term='granizar', translation='grêler', request='verb - formas de género y número')
Output: []

INFLECT(language='English', term='invitation', translation='convocation', request='noun - formas de género y número')
Output: []

"""


PROMPT_PREFIX = """You are a professional linguist specializing in term inflection (including verb conjugation).

Task:
- Output ONLY a Python list containing the correctly inflected forms of the given term.
- The "request" parameter ALWAYS specifies:
  1) the part of speech (POS), and
  2) the exact inflectional forms to generate.
- You MUST strictly follow the requested POS and forms.

Rules:
- Use the provided translation ONLY to disambiguate the meaning of the term.
- If the requested POS/forms does NOT match the term, return an empty list.

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

INFLECT(language='English', term='NASA', translation='NASA', request='noun - singular possessive')
Output: ["NASA's"]

INFLECT(language='Spanish', term='granizar', translation='grêler', request='verb - masc sg, fem sg, masc pl, fem pl')
Output: []

INFLECT(language='English', term='bear', translation='carry', request='verb - simple past')
Output: ['bore']

INFLECT(language='English', term='bear', translation='oso', request='noun - plural')
Output: ['bears']

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
        "noun - number and possessive forms",
        "adj - comparative/superlative forms",
    ],
}


