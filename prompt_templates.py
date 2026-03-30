
PROMPT_TEMPLATE_English = """You are a professional English linguist.

Task:
Generate all standard inflected forms of the given term as a JSON array of strings.

Constraints:
- Only standard morphological inflections of the term.
- Include singular/plural for nouns, gender/case if relevant.
- Include all verb forms (infinitive, tenses, participles, etc.).
- Include irregular forms when they exist.
- Inflect multi-word expressions if applicable.
- If the term is a verb in infinitive form (e.g., "to view"), keep "to" only for the infinitive.
- All other conjugated forms should not include "to".
- Do NOT include:
  - related words, derivations, compounds (e.g., "schematics", "doghouse")
  - possessives ("dog's")
  - comparative/superlative forms ("more beautiful")
  - acronyms or proper nouns (e.g., "UNESCO", "USA", "I.B.M.")
- Use the POS provided to guide inflection.
- Do NOT explain anything or give reasoning.
- Output JSON array only.

Examples:

Input:
put on (verb)

Output:
["put on", "puts on", "putting on"]

Input:
schema (noun)

Output:
["schema", "schemas", "schemata"]

Input:
to view (verb)

Output:
["to view", "view", "views", "viewed", "viewing"]

Input:
{term} ({pos})

Output: </think>
"""

PROMPT_TEMPLATE_French = """You are a professional French linguist.

Task:
Generate all standard inflected forms of the given term as a JSON array of strings.

Constraints:
- Only standard morphological inflections of the term.
- Use the provided part of speech (POS) to guide inflection.

Nouns and adjectives:
- Include all gender and number forms (masculine, feminine, singular, plural).

Verbs:
- Include ALL simple (non-compound) forms:
  - Infinitive
  - Present (all persons)
  - Imperfect (all persons)
  - Simple past (passé simple, all persons)
  - Simple future (all persons)
  - Conditional (all persons)
  - Imperative (2nd person singular/plural, 1st person plural)
  - Present participle
  - Past participle

- Do NOT include:
  - Subjunctive forms
  - Compound tenses (no auxiliary verbs: no "avoir" or "être")
  - Subject pronouns (no "je", "tu", etc.)

General:
- Include irregular forms when they exist.
- Inflect multi-word expressions correctly.
- Do NOT include:
  - derived words, compounds, or related words
  - duplicates

Output format:
- JSON array of strings only
- No explanation, no comments, no extra text

Examples:

Input:
chat (noun)

Output:
["chat", "chats", "chatte", "chattes"]

Input:
manger (verb)

Output:
["manger", "mange", "manges", "mangeons", "mangez", "mangent",
 "mangeais", "mangeait", "mangions", "mangiez", "mangeaient",
 "mangeai", "mangeas", "mangea", "mangeâmes", "mangeâtes", "mangèrent",
 "mangerai", "mangeras", "mangera", "mangerons", "mangerez", "mangeront",
 "mangerais", "mangerait", "mangerions", "mangeriez", "mangeraient",
 "mangeant", "mangé"]

Input:
{term} ({pos})

Output: </think>
"""