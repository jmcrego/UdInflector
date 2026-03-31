
import ast

PROMPT_PREFIX_IDS = None  # will be set after loading tokenizer
PROMPT_PREFIX = """You are a professional linguist specializing in inflection (including verb conjugation).

Task:
- Output ONLY a Python list with conjugated/inflected forms of the term
- Guide your conjugation/inflection based on the part of speech (POS) and language-specific rules for the given term.
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

Input: French verb \'parler\', present indicative
Output: ['parle', 'parles', 'parle', 'parlons', 'parlez', 'parlent']

Input: French verb \'parler\', participe passé
Output: ['parlé', 'parlée', 'parlés', 'parlées']

Input: English noun \'box\'
Output: ['box', 'boxes']

Input: Spanish adjective \'bonito\'
Output: ['bonito', 'bonita', 'bonitos', 'bonitas']

Input: English verb \'to live\', infinitive
Output: ['to live']

Input: English verb \'to speak\', infinitive / gerund / past participle
Output: ['speak', 'speaking', 'spoken']

Input: Spanish verb \'granizar\', imperativo
Output: []

Input: """

TENSES = {
    "French": [
        "infinitif / gérondif / participe",
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
        "infinitivo / gerundio / participio",
        "presente indicativo",
        "pretérito indefinido",
        "imperativo",
        "futuro simple",
        "subjuntivo presente",
        "condicional simple",
        "pretérito indefinido",
        "subjuntivo imperfecto"
    ],
    "English": [
        "base form / 3rd person singular present / past simple / past participle / gerund (-ing form)",
    ],
}

# ------------------------------
# Generate prompt
# ------------------------------
def generate_sample(language: str, pos: str, term: str, ud: str, tokenizer):

    prompts = []

    if pos == "verb":
        for tense in TENSES[language]:
            dynamic_text = f"{language}, {pos}, {term}, {tense}\nOutput:"
            dynamic_text_ids = tokenizer(dynamic_text, return_tensors=None)["input_ids"]
            d = {
                "language": language, 
                "pos": pos, 
                "term": term, 
                "ud": ud,
                "tense": tense, 
                "prompt": PROMPT_PREFIX + dynamic_text,
                "prompt_ids": PROMPT_PREFIX_IDS + dynamic_text_ids,
            }
            prompts.append(d)

    else:
        dynamic_text = f"{language}, {pos}, {term}\nOutput:"
        dynamic_text_ids = tokenizer(dynamic_text, return_tensors=None)["input_ids"]
        d = {
            "language": language, 
            "pos": pos, 
            "term": term, 
            "ud": ud,
            "prompt": PROMPT_PREFIX + dynamic_text,
            "prompt_ids": PROMPT_PREFIX_IDS + dynamic_text_ids
        }
        prompts.append(d)

    return prompts

def filter_list(forms: list[str]) -> list[str]:
    # use final word if multiple words
    # avoid duplicates
    new_forms = []
    for form in forms:
        form = form.strip().split(" ")[-1] 
        if form in new_forms:
            continue
        new_forms.append(form)
    return new_forms


def get_list_from_string(text: str) -> list[str]:
    try:
        data_filtered = text.strip()
        if data_filtered.startswith('[') and data_filtered.endswith(']'):
            data = ast.literal_eval(data_filtered)
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return filter_list(data)
    except Exception as e:
        pass
    return []

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Multilingual Batched Conjugation Generator", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('tsv', type=str, help="Path to TSV file with terms to inflect (columns: term, pos)")
    parser.add_argument('--language', type=str, required=True, choices=['English', 'French', 'Spanish'], help="Language (e.g. English, French, Spanish)")
    parser.add_argument('--out', type=str, default="conjugation_outputs.jsonl", help="Output Jsonl file to save conjugations/inflections")
    parser.add_argument('--model', type=str, default='/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-32B', help="Path to LLM model")
    parser.add_argument('--max_tokens', type=int, default=256, help="Maximum tokens to generate for each prompt (output only)")
    parser.add_argument('--max_model_len', type=int, default=512, help="Maximum sequence length for model input (prompt + output)")
    parser.add_argument('--max_num_seqs', type=int, default=512, help="Maximum number of sequences to generate per prompt (batch size)")
    parser.add_argument('--temperature', type=float, default=0.0, help="Sampling temperature for generation (0 for deterministic)")
    parser.add_argument('--top_p', type=float, default=1.0, help="Nucleus sampling top-p value (1.0 for no nucleus sampling)")
    parser.add_argument('--stop', type=str, nargs='*', default=["\n"], help="List of stop tokens to end generation (use '\\n' for newline)")
    parser.add_argument('--dtype', type=str, default='auto', help="Data type for model (e.g. 'auto', 'float16', 'bfloat16')")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help="Fraction of GPU memory to use (0-1, e.g. 0.95 for 95%)")
    args = parser.parse_args()

    # Load only the LLM tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Tokenize the static prompt prefix once since it's the same for all entries
    PROMPT_PREFIX_IDS = tokenizer(PROMPT_PREFIX, return_tensors=None)["input_ids"]

    prompts = []

    with open(args.tsv, 'r') as f:
        nlines = 0
        for line in f:
            nlines += 1
            term, pos, ud = line.strip().split('\t')
            new_prompts = generate_sample(args.language, pos, term, ud, tokenizer)
            prompts += new_prompts

    print(f"Generated {len(prompts)} prompts from {nlines} glossary lines. Starting generation...")

    #check if running on V100, A100 or H100 and set dtype accordingly
    if args.dtype == 'auto':
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Running on GPU: {gpu_name}")
            if "V100" in gpu_name:
                args.dtype = 'float16'
            elif "A100" in gpu_name or "H100" in gpu_name:
                args.dtype = 'bfloat16'
            else:
                print("Unknown GPU type, defaulting to float16")
                args.dtype = 'float16'
        else:
            print("No GPU detected, defaulting to float16")
            args.dtype = 'float16'

    # generate conjugations/inflections in batches 
    from vllm import LLM, SamplingParams, TokensPrompt

    llm: LLM = LLM(
        model=args.model,
        max_model_len=args.max_model_len, 
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype, 
    )
    print(f"Loaded model {args.model} with dtype {args.dtype}")

    sampling_params: SamplingParams = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
    )

    batch_prompts = [
        TokensPrompt(prompt_token_ids=p["prompt_ids"])
        for p in prompts
    ]
    tic = time.perf_counter()
    outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
    toc = time.perf_counter()
    print(f"Generation completed in {toc - tic:.2f} seconds")

    with open(args.out, "w") as of:
        for i, output in enumerate(outputs):
            print(f"Output[{i}]: {output.outputs[0].text.strip()}")
            prompts[i]["output"] = get_list_from_string(output.outputs[0].text.strip())
            of.write(
                f"idx: {i}, {prompts[i]['language']}, {prompts[i]['pos']}, {prompts[i]['term']}, {prompts[i].get('tense', '-')}\t{prompts[i]['output']}\t{prompts[i]['ud']}\n"
            )


