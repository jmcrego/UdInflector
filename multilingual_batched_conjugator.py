from vllm import LLM, TokenizedPrompt
import ast

# ------------------------------
# 1️⃣ Initialize the model
# ------------------------------
llm = LLM(model="meta-llama/Llama-2-13b-chat-hf")

# ------------------------------
# 2️⃣ Cached prefixes dictionary
# ------------------------------
cached_prefixes: dict[str, TokenizedPrompt] = {}

# ------------------------------
# 3️⃣ Function to get/build prefix for a language
# ------------------------------
def get_prefix(language: str) -> TokenizedPrompt:
    """
    Return a tokenized prefix for the given language.
    Build and cache it if not yet present.
    """
    lang_key = language.lower()
    if lang_key not in cached_prefixes:
        prefix_text = f"""You are a professional multilingual linguist specializing in verb conjugation.

Task:
- Output ONLY a Python dictionary where:
  - keys are tense names
  - values are lists of conjugated forms
- Each list must have the correct number of forms for the tense.
- Use the standard grammatical persons and order for each tense.
- Do NOT include pronouns, explanations, or extra text.
- Ensure correct spelling, accents, and irregular forms.

Example:
Input: present indicative, parler
Output:
{{'present indicative': ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']}}

Now produce the result for the verb:
"""
        cached_prefixes[lang_key] = TokenizedPrompt.from_text(prefix_text, llm.tokenizer)
    return cached_prefixes[lang_key]

# ------------------------------
# 4️⃣ Safe parsing helper
# ------------------------------
def safe_parse_dict(text: str) -> dict[str, list[str]]:
    """
    Safely parse model output into a Python dict of tense -> list of conjugated forms
    """
    try:
        data = ast.literal_eval(text)
        if not isinstance(data, dict):
            raise ValueError("Output is not a dictionary")
        for tense, forms in data.items():
            if not isinstance(forms, list) or not all(isinstance(f, str) for f in forms):
                raise ValueError(f"Invalid forms list for tense '{tense}'")
            if len(forms) == 0:
                raise ValueError(f"Empty forms list for tense '{tense}'")
        return data
    except Exception as e:
        raise ValueError(f"Invalid conjugation output: {text}") from e

# ------------------------------
# 5️⃣ Generate batched conjugations
# ------------------------------
def generate_conjugation_batch(language: str, verb: str, tenses: list[str], max_tokens: int = 256) -> dict[str, list[str]]:
    """
    Generate conjugations for multiple tenses of the same verb in a given language.
    Returns a dictionary: {tense -> list of forms}
    """
    cached_prefix = get_prefix(language)
    
    dynamic_text = f"verb: {verb}\ntenses: {', '.join(tenses)}"
    dynamic_part = TokenizedPrompt.from_text(dynamic_text, llm.tokenizer)
    
    full_prompt = cached_prefix + dynamic_part
    output = llm.generate(full_prompt, max_tokens=max_tokens)
    text = output.text.strip()
    
    return safe_parse_dict(text)

options = {
    "French": {
        "verb": ["infinitif", "présent indicatif", "imparfait indicatif", "impératif", "futur simple", "passé composé", 
                    "subjonctif présent", "conditionnel présent", "plus-que-parfait", "futur antérieur", 
                    "passé simple", "passé antérieur", "subjonctif imparfait", "conditionnel passé"],
        "noun": ["masculin singulier, féminin singulier, masculin pluriel, féminin pluriel"],
        "adj": ["masculin singulier, féminin singulier, masculin pluriel, féminin pluriel, superlatif, comparatif"]
    },
    "Spanish": {
        "verb": ["infinitivo", "presente indicativo", "pretérito indicativo", "imperativo", "futuro simple", 
                    "pretérito perfecto compuesto", "subjuntivo presente", "condicional simple", "pretérito pluscuamperfecto", 
                    "futuro perfecto", "pretérito indefinido", "pretérito anterior", "subjuntivo imperfecto", "condicional compuesto"],
        "noun": ["masculino singular, femenino singular, masculino plural, femenino plural"],
        "adj": ["masculino singular, femenino singular, masculino plural, femenino plural, superlativo, comparativo"]
    },
    "English": {
        "verb": ["infinitive", "present simple", "past simple", "imperative", "future simple", "present perfect", 
                    "present subjunctive", "conditional simple", "past perfect", "future perfect", 
                    "past simple (literary)", "past perfect (literary)", "past subjunctive", "conditional perfect"],
        "noun": ["singular, plural"],
        "adj": ["positive, comparative, superlative"]
    },
}

# ------------------------------
# 6️⃣ Example usage
# ------------------------------
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Multilingual Batched Conjugation Generator", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--language', type=str, required=True, help="Language for conjugation (e.g. 'French', 'Spanish', 'English')")
    parser.add_argument('--term', type=str, required=True, help="Verb to conjugate (e.g. 'parler', 'hablar', 'to speak')")
    parser.add_argument('--pos', type=str, required=True, choices=['verb', 'noun', 'adj'], help="Part of speech (either 'verb', 'noun' or 'adj')")
    args = parser.parse_args()
    # French

    batch = generate_conjugation_batch(args.language, args.term, args.pos, options[args.language][args.pos])
    print(batch)


