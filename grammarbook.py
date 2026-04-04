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
MODEL_PATH = "/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-VL-32B-Instruct"
MODEL_PATH = "/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen2.5-VL-32B-Instruct"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90
DTYPE = "auto"
MAX_TOKENS = 4000
GEN_BATCH_SIZE = 2
MAX_IMAGE_SIDE = 1024

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
PROMPT = """
You are given one image corresponding to one page of a French grammar book.

Your task is to copy the content into a raw text file, without adding any interpretation, formatting, or summarization.

GENERAL INSTRUCTIONS:
- Convert ALL visible content into plain raw text.
- Do NOT add any content that is not explicitly present in the images.
- Do NOT omit any content and use the EXACT wording from the images.
- The final output must be fully understandable by a human reader.
- Preserve the original document order exactly.

TEXT PRESERVATION:
- Preserve all text exactly as it appears in the images.
- Preserve all examples, exercises, and explanations in their entirety.
- Only correct obvious OCR errors (e.g., broken or split words).
- Do NOT rewrite, summarize, or interpret the text.

FORMATTING RULES:
- Do NOT use any markdown, HTML, or special formatting in the output.
- If a word/phrase is emphasized in the original text, wrap it in [brackets].

NON-TEXT ELEMENTS:
- Tables → use plain text with columns separated by "|"
- Lists in multiple columns → preserve the order and structure in plain text using bullet points "-" or numbering "i."
- Figures and diagrams → describe them clearly
- Prefer clarity and readability over verbatim reproduction of non-text elements.

CLEANING RULES:
Remove:
- page numbers
- headers/footers
- references like "see page X"

OUTPUT:
- Only raw text
"""

# =========================
# STEP 4: BUILD MM PROMPTS + BATCH GENERATE
# =========================
def build_mm_prompt(image_path: str):
    # Load and preprocess images
    images = load_images([image_path])
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    image_placeholders = image_token
    prompt_text = f"{image_placeholders}\n\n{PROMPT}" if images else PROMPT

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
        mm_prompts = [build_mm_prompt(page_path) for page_path in page_paths[start:end]]
        outputs = llm.generate(mm_prompts, sampling_params=sampling_params)
        for i, out in enumerate(outputs):
            page_idx = start + i
            txt_path = page_paths[page_idx].replace(".png", f"{os.path.basename(MODEL_PATH)}.txt")
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