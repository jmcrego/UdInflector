
import ast

PROMPT_PREFIX_IDS = None  # will be set after loading tokenizer
PROMPT_PREFIX = """You are a professional linguist specializing in conjugation/inflection.

Task:
- Output ONLY a Python list with conjugated/inflected forms of the term given its Part-Of-Speech (POS) and language.
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
def generate_prompts(language: str, pos: str, term: str, tokenizer):

    prompts = []

    if pos == "verb":
        for tense in TENSES[language]:
            dynamic_text = f"{language}, {pos}, {term}, {tense}\nOutput:"
            dynamic_text_ids = tokenizer(dynamic_text, return_tensors=None)["input_ids"]
            d = {
                "language": language, 
                "pos": pos, 
                "term": term, 
                "tense": tense, 
                "prompt": PROMPT_PREFIX + dynamic_text,
                "prompt_ids": PROMPT_PREFIX_IDS + dynamic_text_ids
            }
            prompts.append(d)

    else:
        dynamic_text = f"{language}, {pos}, {term}\nOutput:"
        dynamic_text_ids = tokenizer(dynamic_text, return_tensors=None)["input_ids"]
        d = {
            "language": language, 
            "pos": pos, 
            "term": term, 
            "prompt": PROMPT_PREFIX + dynamic_text,
            "prompt_ids": PROMPT_PREFIX_IDS + dynamic_text_ids
        }
        prompts.append(d)

    return prompts

def get_list_from_string(text: str) -> list[str]:
    try:
        data_filtered = text.strip()
        if data_filtered.startswith('[') and data_filtered.endswith(']'):
            data = ast.literal_eval(data_filtered)
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return data
    except Exception as e:
        pass
    return []

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
    if args.term and args.pos:
        prompts += generate_prompts(args.language, args.pos, args.term, tokenizer)

    if args.tsv:
        with open(args.tsv, 'r') as f:
            for line in f:
                term, pos = line.strip().split('\t')
                prompts += generate_prompts(args.language, pos, term, tokenizer)


    print(f"Generated {len(prompts)} prompts. Starting generation...")

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
    outputs = llm.generate(batch_prompts, sampling_params=sampling_params)

    with open(args.out, "w") as of:
        for i, output in enumerate(outputs):
            prompts[i]["output"] = get_list_from_string(output.outputs[0].text.strip())
            of.write(f"idx: {i}, language: {prompts[i]['language']} ({prompts[i]['pos']}) {prompts[i]['term']} {prompts[i].get('tense', '')} => {prompts[i]['output']}\n")
            of.flush()


