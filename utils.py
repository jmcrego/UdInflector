
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
- Preserve a consistent, linguistically standard ordering of forms.
- Only include valid, attested inflected forms (including irregular ones).
- Do NOT include synonyms, derivations, or related words.
- For multi-word expressions, inflect the full expression appropriately.
- Use standard modern orthography for the language.

Output format:
- Return ONLY a Python list of strings.
- No explanations, comments, or additional text.

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


REQUESTS = {
    "French": [
        "verb - infinitif",
        "verb - participe présent",
        "verb - participe passé",
        "verb - présent indicatif",
        "verb - imparfait indicatif",
        "verb - impératif",
        "verb - futur simple",
        "verb - subjonctif présent",
        "verb - conditionnel présent",
        "verb - passé simple",
        "verb - subjonctif imparfait",
        "noun - formes de genre et de nombre",
        "adj - formes de genre et de nombre",
    ],
    "Spanish": [
        "verb - infinitivo",
        "verb - gerundio", 
        "verb - participio",
        "verb - presente indicativo",
        "verb - pretérito indefinido",
        "verb - imperativo",
        "verb - futuro simple",
        "verb - subjuntivo presente",
        "verb - condicional simple",
        "verb - subjuntivo imperfecto",
        "noun - formas de género y número",
        "adj - formas de género y número",
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


