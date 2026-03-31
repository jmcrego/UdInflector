
import ast
from utils import PROMPT_PREFIX, INFLECTIONS, get_dtype_for_gpu

def generate_sample(language: str, pos: str, term: str, ud: str, tokenizer, prompt_prefix_ids):

    prompts = []
    if pos in INFLECTIONS and language in INFLECTIONS[pos]:
         for inflection in INFLECTIONS[pos][language]:
            dynamic_text = f"Input: {language} {pos} '{term}'\nForms: {inflection}\nOutput:"
            dynamic_text_ids = tokenizer(dynamic_text, return_tensors=None)["input_ids"]
            d = {
                "language": language, 
                "pos": pos, 
                "term": term, 
                "ud": ud,
                "inflection": inflection, 
                "prompt": PROMPT_PREFIX + dynamic_text,
                "prompt_ids": prompt_prefix_ids + dynamic_text_ids,
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
    parser.add_argument('--out', type=str, default=None, help="Output Jsonl file to save conjugations/inflections")
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

    if args.out is None:
        args.out = f"{args.tsv}.inflections.tsv"    

    # Load only the LLM tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Tokenize the static prompt prefix once since it's the same for all entries
    PROMPT_PREFIX_IDS = tokenizer(PROMPT_PREFIX, return_tensors=None)["input_ids"]

    samples = []

    with open(args.tsv, 'r') as f:
        nlines = 0
        for line in f:
            nlines += 1
            term, pos, ud = line.strip().split('\t')
            new_prompts = generate_sample(args.language, pos, term, ud, tokenizer, PROMPT_PREFIX_IDS)
            samples += new_prompts

    print(f"Generated {len(samples)} prompts from {nlines} glossary lines. Starting generation...")

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
            output_list = get_list_from_string(output.outputs[0].text.strip())
            if len(output_list) == 0:
                output_list = [sample['term']]
            request = f"{i} {sample['language']}, {sample['pos']}, {sample['term']}, {sample['inflection']}"
            of.write(f"{sample['ud']}\t{output_list}\t{request}\n")
            print(f"{sample['ud']}\t{output_list}\t{request}")

