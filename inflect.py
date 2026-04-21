
import ast
from utils import PROMPT_PREFIX, EXAMPLES, REQUESTS, get_dtype_for_gpu

def read_tsv(path: str, language: str, prompt_prefix_ids: list[int], tokenizer) -> list[dict]:
    samples = []
    with open(path, 'r') as f:
        nlines = 0
        for line in f:
            nlines += 1
            term1, term2 = line.strip().split('\t')
            new_samples = generate_sample(language, term1, term2, prompt_prefix_ids, tokenizer)
            samples += new_samples
    print(f"Generated {len(samples)} prompts from {nlines} UD pairs.")
    return samples

def generate_sample(language: str, term1: str, term2: str, prompt_prefix_ids: list[int], tokenizer) -> list[dict]:
    prompts = []
    if language in REQUESTS:
         for request in REQUESTS[language]:
            pos, request = request.split(" - ")
            dynamic_text = f"INFLECT(language='{language}', term='{term1}', translation='{term2}', pos='{pos}', request='{request}')\nOutput:"
            dynamic_text_ids = tokenizer(dynamic_text, return_tensors=None)["input_ids"]
            d = {
                "language": language,
                "term": term1,
                "translation": term2,
                "pos": pos,
                "request": request,
                "prompt_ids": prompt_prefix_ids + dynamic_text_ids,
            }
            prompts.append(d)
    return prompts


def filter_list(forms: list[str]) -> list[str]:
    new_forms = []
    for form in forms:
        # normalize whitespace
        form = " ".join(form.strip().split()) 
        # comparative/superlative correction # This is ugly... should be handled in the prompt/request design instead
        if form.startswith("more "):
            form = form[5:]
        if form.startswith("most "):
            form = form[5:]
        # Avoid empty forms
        if len(form) == 0:
            continue
        # Avoid duplicates while preserving order
        if form in new_forms:
            continue
        new_forms.append(form)
    return new_forms


def parse_output(text: str) -> str:
    try:
        data_filtered = text.strip()
        if data_filtered.startswith('[') and data_filtered.endswith(']'):
            data = ast.literal_eval(data_filtered)
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return ';'.join(filter_list(data))
    except Exception as e:
        pass
    print(f"Output is not a valid list: {text}")
    return ""

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Multilingual Batched Conjugation Generator", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('tsv', type=str, help="Path to TSV file with terms to inflect (columns: term, pos)")
    parser.add_argument('--language', type=str, required=True, choices=['English', 'French', 'Spanish'], help="Language (e.g. English, French, Spanish)")
    parser.add_argument('--out', type=str, required=True, help="Output JSONL file to save conjugations/inflections")
    parser.add_argument('--model', type=str, default='/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-32B', help="Path to LLM model")
    parser.add_argument('--max_tokens', type=int, default=256, help="Maximum tokens to generate for each prompt (output only)")
    parser.add_argument('--max_model_len', type=int, default=2048, help="Maximum sequence length for model input (prompt + output)")
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
    PROMPT_PREFIX_IDS = tokenizer(PROMPT_PREFIX + EXAMPLES[args.language], return_tensors=None)["input_ids"]
    print(f"Tokenized PROMPT_PREFIX_IDS into {len(PROMPT_PREFIX_IDS)} tokens")

    samples = read_tsv(args.tsv, language=args.language, prompt_prefix_ids=PROMPT_PREFIX_IDS, tokenizer=tokenizer)

    #check if running on V100, A100 or H100 and set dtype accordingly    
    if args.dtype == 'auto':
        args.dtype = get_dtype_for_gpu()

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

    batch_prompts = [TokensPrompt(prompt_token_ids=p["prompt_ids"]) for p in samples]

    tic = time.perf_counter()
    outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
    print(f"Generation completed in {time.perf_counter() - tic:.2f} seconds")

    with open(args.out, "w") as of:
        for i, (sample, output) in enumerate(zip(samples, outputs)):
            ud = sample['term'] + " ➤ " + sample['translation']
            req = sample['request']
            pos = sample['pos']
            out = parse_output(output.outputs[0].text.strip())
            of.write(f"{ud}\t{out}\t{pos} - {req}\n")

