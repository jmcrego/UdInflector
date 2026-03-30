
from transformers import AutoTokenizer

PROMPT_PREFIX = """You are a professional linguist specializing in conjugation/inflection.

Task:
- Output ONLY a Python list with conjugated/inflected forms of the term given its Part-Of-Speech (POS).
- For verbs:
  * Provide the correct number of form conjugations for the given tense.
  * Provide all inflection combinations when applicable (masculine, feminine, singular, plural).
- For nouns/adjectives:
  * Provide all inflection combinations when applicable (masculine/feminine, singular/plural).
- Do NOT include pronouns, explanations, or extra text.
- Ensure correct spelling, accents, and irregular forms.
- Follow the standard order of forms for each language and POS.
- If the term is not inflectable or the tense is not applicable, return an empty list.

Examples:

Input: French, verb, parler, present indicative
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

Input: French, verb, parler, participe passé
Output: ['parlé', 'parlée', 'parlés', 'parlées']

Input: English, noun, box
Output: ['box', 'boxes']

Input: Spanish, adj, bonito, adj
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

Input: English, verb, to live, infinitive
Output: ['to live']

Input: Spanish, verb, granizar, imperativo
Output: []

Input: """

TENSES = {
    "French": ["infinitif", "présent indicatif", "imparfait indicatif", "impératif", "futur simple", "passé composé", 
                "subjonctif présent", "conditionnel présent", "plus-que-parfait", "futur antérieur", 
                "passé simple", "passé antérieur", "subjonctif imparfait", "conditionnel passé"],
    "Spanish": ["infinitivo", "presente indicativo", "pretérito indicativo", "imperativo", "futuro simple", 
                "pretérito perfecto compuesto", "subjuntivo presente", "condicional simple", "pretérito pluscuamperfecto", 
                "futuro perfecto", "pretérito indefinido", "pretérito anterior", "subjuntivo imperfecto", "condicional compuesto"],
    "English": ["infinitive", "present simple", "past simple", "imperative", "future simple", "present perfect", 
                "present subjunctive", "conditional simple", "past perfect", "future perfect", 
                "past simple (literary)", "past perfect (literary)", "past subjunctive", "conditional perfect"],
}

# ------------------------------
# Generate prompt
# ------------------------------
def generate_prompts(PROMPT_PREFIX_IDS, language: str, pos: str, term: str, tokenizer):

    prompts = []

    if pos == "verb":
        for tense in TENSES[language]:
            dynamic_text = f"{language}, {pos}, {term}, {tense}\nOutput:"
            dynamic_text_ids = tokenizer(dynamic_text, return_tensors=None)["input_ids"]
            prompts.append(PROMPT_PREFIX_IDS + dynamic_text_ids)
            print(f"{dynamic_text}\n{dynamic_text_ids}")

    else:
        dynamic_text = f"{language}, {pos}, {term}\nOutput:"
        dynamic_text_ids = tokenizer(dynamic_text, return_tensors=None)["input_ids"]
        prompts.append(PROMPT_PREFIX_IDS + dynamic_text_ids)
        print(f"{dynamic_text}\n{dynamic_text_ids}")

    return prompts


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multilingual Batched Conjugation Generator", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--language', type=str, required=True, choices=['English', 'French', 'Spanish'], help="Language (e.g. English, French, Spanish)")
    parser.add_argument('--term', type=str, required=False, help="Term to inflect (e.g. 'parler', 'box', 'bonito')")
    parser.add_argument('--pos', type=str, required=False, choices=['verb', 'noun', 'adj'], help="Part of speech (e.g. 'verb', 'noun', 'adj')")
    parser.add_argument('--tsv', type=str, required=False, help="Path to TSV file with terms to inflect (columns: term, pos)")
    parser.add_argument('--model', type=str, default='/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-32B', help="Path to LLM model")
    args = parser.parse_args()

    # Load only the LLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Tokenize the static prompt prefix once since it's the same for all entries
    PROMPT_PREFIX_IDS = tokenizer(PROMPT_PREFIX, return_tensors=None)["input_ids"]

    prompts = []
    if args.term and args.pos:
        prompts += generate_prompts(PROMPT_PREFIX_IDS, args.language, args.pos, args.term, tokenizer)

    if args.tsv:
        with open(args.tsv, 'r') as f:
            for line in f:
                term, pos = line.strip().split('\t')
                prompts += generate_prompts(PROMPT_PREFIX_IDS, args.language, pos, term, tokenizer)

    # generate conjugations/inflections in batches 
    # from vllm import LLM
    # llm: LLM = LLM(model=args.model)
