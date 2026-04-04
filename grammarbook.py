import fitz  # PyMuPDF
import os
import sys
from typing import List
from PIL import Image
from vllm import LLM, SamplingParams

if len(sys.argv) != 2:
    print("Usage: python grammarbook.py <path_to_pdf>")
    sys.exit(1)

# =========================
# CONFIG
# =========================
PDF_PATH = sys.argv[1]  # Path to input PDF
OUT_PATH = PDF_PATH.replace(".pdf", ".txt")
OUT_DIR = PDF_PATH.replace(".pdf", "_images")
DPI = 200

# vLLM settings
MODEL_PATH = "/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen2.5-VL-32B-Instruct"
# MODEL_PATH = "/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-VL-32B-Instruct"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.98
DTYPE = "auto"
MAX_TOKENS = 2048
GEN_BATCH_SIZE = 4
MAX_IMAGE_SIDE = 2048

# =========================
# STEP 1: PDF → IMAGES
# =========================
def pdf_to_images(pdf_path, output_dir, dpi=200):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        path = os.path.join(output_dir, f"page_{i:04d}.png")
        pix.save(path)
        paths.append(path)

    return paths

# =========================
# STEP 2: IMAGE → PIL
# =========================
def load_images(paths: List[str]):
    images = []
    for path in paths:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            # Downscale to keep visual token count under max_model_len.
            rgb.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            images.append(rgb)
    return images

# =========================
# STEP 3: PROMPT
# =========================
PROMPT1 = """
You are given one image corresponding to one page of a French grammar book.

Your task is to transcribe ALL visible text into plain text, with no interpretation, summarization, or reformulation.

GENERAL INSTRUCTIONS:
- Transcribe ALL visible text exactly as it appears, word for word.
- Do NOT add, remove, or rephrase anything.
- Preserve the original reading order: for two-column layouts, read the left column top-to-bottom first, then the right column.
- The output must be fully readable as a standalone text.

FRENCH LANGUAGE:
- Preserve all French accented characters exactly: é, è, ê, ë, à, â, ä, ù, û, ü, ô, ö, î, ï, ç, œ, æ and their uppercase variants.
- Do NOT replace accented characters with unaccented ones (e.g., never write "e" for "é").
- Preserve typographic apostrophes and quotation marks (« », ‹ ›) as they appear.

GRAMMAR BOOK STRUCTURE:
- Entry headwords (usually bold or small caps): transcribe them in UPPERCASE.
- Example sentences (usually in italics): transcribe them normally, preceded by a dash "- ".
- Grammatical labels and abbreviations (n., v., adj., etc.): transcribe exactly as shown.
- Conjugation tables and paradigm grids: reproduce using plain text with columns separated by "|".
- Numbered or lettered lists: preserve the numbering exactly.

FORMATTING RULES:
- Do NOT use markdown, HTML, or special symbols not present in the original.
- Do NOT add line breaks that are not present in the original text flow.
- Page numbers and running headers/footers: skip them silently.

OUTPUT:
- Only the transcribed page text, nothing else.
"""

PROMPT = """
You are given one image corresponding to one page of a French grammar book.

Your task is to transcribe ALL visible text into plain text, with no interpretation, summarization, or reformulation.

GENERAL INSTRUCTIONS:
- Transcribe ALL visible text exactly as it appears, word for word.
- Do NOT add, remove, or rephrase anything.
- Remove silently any page numbers, running headers/footers, or text that is not part of the main content.
- Preserve the original reading order: for two-column layouts, transcribe the left column first, then the right column.
- The output must be fully readable as a standalone text.

FRENCH LANGUAGE:
- Preserve all French accented characters exactly.
- Do NOT replace accented characters with unaccented ones.
- Preserve typographic apostrophes and quotation marks (« », ‹ ›) as they appear.

GRAMMAR BOOK STRUCTURE:
- Example sentences (usually in italics): transcribe them normally.
- Conjugation tables and paradigm grids: Convert them into explicit, linear text. For example:
    indicatif présent: je conduis; tu conduis; il/elle conduit; nous conduisons; vous conduisez; ils/elles conduisent
- Numbered or lettered lists: preserve the numbering exactly.
- Figures and diagrams: if they contain visible text, transcribe that text in the correct reading order.

FORMATTING RULES:
- Do NOT use markdown, HTML, or special symbols not present in the original.
- Do NOT add line breaks that are not present in the original text flow.
- Page numbers and running headers/footers: skip them silently.

OUTPUT:
- Only the transcribed page text, nothing else.
"""

# =========================
# STEP 4: BUILD MM PROMPTS + BATCH GENERATE
# =========================
def build_mm_prompt(image_path: str, tokenizer):
    images = load_images([image_path])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": prompt_text,
        "multi_modal_data": {"image": images},
    }


def generate_pages_batched(llm: LLM, page_paths: List[str]):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0, #deterministic
        top_p=1.0, #no nucleus sampling
        max_tokens=MAX_TOKENS, #only output tokens, not prompt tokens
    )
    for start in range(0, len(page_paths), GEN_BATCH_SIZE):
        end = min(start + GEN_BATCH_SIZE, len(page_paths))
        print(f"🤖 Inference mini-batch with pages {start + 1}-{end}/{len(page_paths)}")
        mm_prompts = [build_mm_prompt(page_path, tokenizer) for page_path in page_paths[start:end]]
        outputs = llm.generate(mm_prompts, sampling_params=sampling_params)
        for i, out in enumerate(outputs):
            page_idx = start + i
            txt_path = page_paths[page_idx].replace(".png", f".{os.path.basename(MODEL_PATH)}.txt")
            generated_token_ids = out.outputs[0].token_ids
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

# =========================
# MAIN PIPELINE
# =========================
def main():
    print(f"Loading local model with vLLM: {MODEL_PATH}")
    try:
        llm = LLM(
            model=MODEL_PATH,
            dtype=DTYPE,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            max_num_seqs=GEN_BATCH_SIZE,
        )
    except ValueError as e:
        if "model type `qwen3_vl`" in str(e):
            raise SystemExit(
                "Your transformers/vLLM stack does not support qwen3_vl yet. "
                "Use a Qwen2.5-VL checkpoint (default in this script), or upgrade both "
                "transformers and vLLM to versions that support Qwen3-VL."
            ) from e
        raise

    print("Converting PDF to images...")
    images = pdf_to_images(PDF_PATH, OUT_DIR, DPI)

    print(f"Running batched inference for {len(images)} pages...")
    generate_pages_batched(llm, images)

    print(f"Done → per-page text files in {OUT_DIR}/page_####.txt")


if __name__ == "__main__":
    main()