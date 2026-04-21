
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

TASK:
- Given:
  1) a <term>,
  2) a <language> (which determines the inflectional system and resolves cross-linguistic ambiguity),
  3) a <translation> (used only to disambiguate meaning),
  4) and a <request> specifying the part of speech and the exact inflectional paradigm to produce,
- Generate the requested inflected forms of the term.

RULES:
- The <request> ALWAYS defines the part of speech (POS) and the exact inflectional paradigm to produce.
- You MUST strictly follow the requested POS and paradigm.
- If the term does not exist in the given language as a lemma of the requested POS, return an empty list.
- If the requested inflectional paradigm is not fully applicable to the term in the given language, return an empty list.
- Do NOT derive, reinterpret, or convert the term into another part of speech.
- Use the language to determine the inflectional system and resolve ambiguity.
- Use the translation only to disambiguate meaning.
- Do NOT return partial results.

INFLECTION CONSTRAINTS:
- Only include valid, attested inflected forms (including irregular ones).
- Preserve a consistent, linguistically standard ordering of forms.
- Do NOT include synonyms, derivations, or related words.
- For multi-word expressions, inflect the full expression appropriately.
- Use standard modern orthography for the language.

OUTPUT FORMAT:
- Return ONLY a valid Python list of strings, NOTHING else.
"""

EXAMPLES = {
    "French": """
EXAMPLES:

INFLECT(language='French', term='parler', translation='speak' , request='verb - présent indicatif (1s, 2s, 3s, 1p, 2p, 3p)')
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

INFLECT(language='French', term='parler', translation='speak', request='verb - participe passé (masc sg, fem sg, masc pl, fem pl)')
Output: ['parlé', 'parlée', 'parlés', 'parlées']

INFLECT(language='French', term='grouper', translation='juntar', request='verb - subjonctif imparfait (1s, 2s, 3s, 1p, 2p, 3p)')
Output: ['groupasse', 'groupasses', 'groupât', 'groupassions', 'groupassiez', 'groupassent']

INFLECT(language='French', term='être', translation='be', request='verb - présent indicatif (1s, 2s, 3s, 1p, 2p, 3p)')
Output: ['suis', 'es', 'est', 'sommes', 'êtes', 'sont']

INFLECT(language='French', term='aller', translation='go', request='verb - futur simple (1s, 2s, 3s, 1p, 2p, 3p)')
Output: ['irai', 'iras', 'ira', 'irons', 'irez', 'iront']

INFLECT(language='French', term='beau', translation='beautiful', request='adj - masc sg, fem sg, masc pl, fem pl')
Output: ['beau', 'belle', 'beaux', 'belles']

INFLECT(language='French', term='chat', translation='cat', request='noun - masc sg, fem sg, masc pl, fem pl')
Output: ['chat', 'chatte', 'chats', 'chattes']

INFLECT(language='French', term='bouteille', translation='bottle', request='adj - masc sg, fem sg, masc pl, fem pl')
Output: []
""",

    "English": """
EXAMPLES:

INFLECT(language='English', term='box', translation='caja', request='noun - singular, plural')
Output: ['box', 'boxes']

INFLECT(language='English', term='box', translation='caja', request='noun - singular possessive, plural possessive')
Output: ["box's", "boxes'"]

INFLECT(language='English', term='big', translation='grande', request='adj - comparative/superlative forms')
Output: ['big', 'bigger', 'biggest']

INFLECT(language='English', term='speak', translation='hablar', request='verb - base form')
Output: ['speak']

INFLECT(language='English', term='speak', translation='hablar', request='verb - present participle (-ing)')
Output: ['speaking']

INFLECT(language='English', term='go back', translation='volver', request='verb - simple past')
Output: ['went back']

INFLECT(language='English', term='NASA', translation='NASA', request='noun - singular, plural')
Output: ['NASA']

INFLECT(language='English', term='NASA', translation='NASA', request='noun - singular possessive, plural possessive')
Output: ["NASA's"]

INFLECT(language='English', term='bear', translation='llevar', request='verb - simple past')
Output: ['bore']

INFLECT(language='English', term='bear', translation='oso', request='noun - masc sg, fem sg, masc pl, fem pl')
Output: ['bear', 'bear', 'bears', 'bears']

INFLECT(language='English', term='bear', translation='oso', request='verb - past participle')
Output: []

INFLECT(language='English', term='learn', translation='aprender', request='verb - past participle')
Output: ['learned', 'learnt']
""",

    "Spanish": """
EXAMPLES:

INFLECT(language='Spanish', term='bonito', translation='beau', request='adj - masc sg, fem sg, masc pl, fem pl')
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

INFLECT(language='Spanish', term='granizar', translation='grêler', request='noun - masc sg, fem sg, masc pl, fem pl')
Output: []

INFLECT(language='Spanish', term='hablar', translation='speak', request='verb - presente indicativo (1s, 2s, 3s, 1p, 2p, 3p)')
Output: ['hablo', 'hablas', 'habla', 'hablamos', 'habláis', 'hablan']

INFLECT(language='Spanish', term='ir', translation='go', request='verb - futuro simple (1s, 2s, 3s, 1p, 2p, 3p)')
Output: ['iré', 'irás', 'irá', 'iremos', 'iréis', 'irán']

INFLECT(language='Spanish', term='hablar', translation='speak', request='verb - participio (masc sg, fem sg, masc pl, fem pl)')
Output: ['hablado', 'hablada', 'hablados', 'habladas']

INFLECT(language='Spanish', term='comer', translation='eat', request='verb - gerundio')
Output: ['comiendo']

INFLECT(language='Spanish', term='casa', translation='house', request='noun - masc sg, fem sg, masc pl, fem pl')
Output: ['casa', 'casa', 'casas', 'casas']

INFLECT(language='Spanish', term='rápidamente', translation='quickly', request='adj - masc sg, fem sg, masc pl, fem pl')
Output: []
""",

    "Russian": """
EXAMPLES:

INFLECT(language='Russian', term='говорить', translation='speak', request='verb - present (1s, 2s, 3s, 1p, 2p, 3p)')
Output: ['говорю', 'говоришь', 'говорит', 'говорим', 'говорите', 'говорят']

INFLECT(language='Russian', term='писать', translation='write', request='verb - imperative (2s, 2p)')
Output: ['пиши', 'пишите']

INFLECT(language='Russian', term='читать', translation='read', request='noun - nominative sg, nominative pl')
Output: []

INFLECT(language='Russian', term='делать', translation='do', request='verb - past indicative (masc sg, fem sg, neuter sg, plural)')
Output: ['делал', 'делала', 'делало', 'делали']

INFLECT(language='Russian', term='прочитать', translation='read (perfective)', request='verb - past active participle (masc sg, fem sg, neuter sg, plural)')
Output: ['прочитавший', 'прочитавшая', 'прочитавшее', 'прочитавшие']

INFLECT(language='Russian', term='новый', translation='new', request='adj - masc sg, fem sg, neuter sg, plural (nominative)')
Output: ['новый', 'новая', 'новое', 'новые']""",
}


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
    "Russian": [
        "verb - infinitive",
        "verb - adverbial participle",
        "verb - past indicative (masc sg, fem sg, neuter sg, plural)",
        "verb - present (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - imperative (2s, 2p)",
        "verb - future (1s, 2s, 3s, 1p, 2p, 3p)",
        "verb - past active participle (masc sg, fem sg, neuter sg, plural)",
        "verb - past passive participle (masc sg, fem sg, neuter sg, plural)",
        "noun - nominative sg, nominative pl",
        "adj - masc sg, fem sg, neuter sg, plural (nominative)",
    ],
    "English": [
        "verb - base form", 
        "verb - 3rd person singular present", 
        "verb - simple past", 
        "verb - past participle", 
        "verb - present participle (-ing)",
        "noun - singular, plural",
        "noun - singular possessive, plural possessive",
        "adj - comparative/superlative forms",
    ],
}


