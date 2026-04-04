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
OUTPUT_DIR = PDF_PATH.replace(".pdf", "_images")
CHUNK_SIZE = 5
OVERLAP = 1
DPI = 200

# vLLM settings
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen2.5-VL-7B-Instruct",
)
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90
DTYPE = "auto"
MAX_TOKENS = 4000
GEN_BATCH_SIZE = 4

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
# STEP 2: CHUNKING
# =========================
def chunk_with_overlap(items: List[str], chunk_size: int, overlap: int):
    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(items), step):
        chunk = items[i:i + chunk_size]
        if not chunk:
            continue
        chunks.append(chunk)

    return chunks

# =========================
# STEP 3: IMAGE → PIL
# =========================
def load_images(paths: List[str]):
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images

# =========================
# STEP 4: PROMPT
# =========================
PROMPT = """
You are given a sequence of images corresponding to consecutive pages of a French grammar book.

Your task is to copy the content into a raw text file, without adding any interpretation, formatting, or summarization.

GENERAL INSTRUCTIONS:
- Convert ALL visible content into plain raw text.
- Do NOT add any content that is not explicitly present in the images.
- Do NOT omit any content and use the EXACT wording from the images.
- The final output must be fully understandable by a human reader.
- Preserve the original document order exactly.

PAGE STRUCTURE:
- Indicate the beginning of each page with the tag:
  <PAGE:N>
- N must correspond to the page number within this chunk (starting from 1).

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
# STEP 5: BUILD MM PROMPTS + BATCH GENERATE
# =========================
def build_mm_prompt(image_paths: List[str]):
    images = load_images(image_paths)
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    image_placeholders = "\n".join([image_token for _ in images])
    prompt_text = f"{image_placeholders}\n\n{PROMPT}" if images else PROMPT

    return {
        "prompt": prompt_text,
        "multi_modal_data": {"image": images},
    }


def generate_chunks_batched(llm: LLM, chunks: List[List[str]]):
    sampling_params = SamplingParams(
        temperature=0.0, #deterministic
        top_p=1.0, #no nucleus sampling
        max_tokens=MAX_TOKENS, #only output tokens, not prompt tokens
    )
    all_texts = []
    for start in range(0, len(chunks), GEN_BATCH_SIZE):
        end = min(start + GEN_BATCH_SIZE, len(chunks))
        print(f"🤖 Inference mini-batch with chunks {start + 1}-{end}/{len(chunks)}")
        mm_prompts = [build_mm_prompt(chunk) for chunk in chunks[start:end]]
        outputs = llm.generate(mm_prompts, sampling_params=sampling_params)
        all_texts.extend([out.outputs[0].text for out in outputs])

    return all_texts

# =========================
# STEP 6: SPLIT OUTPUT INTO PAGES
# =========================
def split_pages(text):
    pages = {}
    current_page = None
    buffer = []

    for line in text.splitlines():
        if line.startswith("<PAGE:"):
            if current_page is not None:
                pages[current_page] = "\n".join(buffer).strip()
            current_page = int(line.replace("<PAGE:", "").replace(">", "").strip())
            buffer = []
        else:
            buffer.append(line)

    if current_page is not None:
        pages[current_page] = "\n".join(buffer).strip()

    return pages

# =========================
# STEP 7: DEDUPLICATION
# =========================
def merge_chunks(chunk_outputs):
    final_pages = {}

    global_page_index = 0

    for chunk_id, chunk_text in enumerate(chunk_outputs):
        pages = split_pages(chunk_text)

        for local_page, content in pages.items():
            # Compute global page index
            global_idx = chunk_id * (CHUNK_SIZE - OVERLAP) + (local_page - 1)

            if global_idx not in final_pages:
                final_pages[global_idx] = content
            else:
                # Deduplicate: keep longest version
                if len(content) > len(final_pages[global_idx]):
                    final_pages[global_idx] = content

    # Sort pages
    ordered = [final_pages[i] for i in sorted(final_pages.keys())]

    # Rebuild final text
    result = []
    for i, page in enumerate(ordered):
        result.append(f"<PAGE: {i+1}>")
        result.append(page)

    return "\n\n".join(result)

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

    print("📄 Converting PDF to images...")
    images = pdf_to_images(PDF_PATH, OUTPUT_DIR, DPI)

    print("🧩 Creating chunks...")
    chunks = chunk_with_overlap(images, CHUNK_SIZE, OVERLAP)

    print(f"Total chunks: {len(chunks)}")

    print(f"🤖 Running batched inference for {len(chunks)} chunks...")
    outputs = generate_chunks_batched(llm, chunks)

    print("🔁 Merging & deduplicating...")
    final_text = merge_chunks(outputs)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"✅ Done → {OUT_PATH}")


if __name__ == "__main__":
    main()