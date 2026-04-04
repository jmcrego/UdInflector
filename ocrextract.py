"""Extract text from a PDF using PaddleOCR.

Usage:
    python ocrextract.py <input_pdf> [--dpi 150] [--lang fr] [--fast] [--max-pages 1]
"""

import argparse
import os
from typing import List, Optional

import fitz  # PyMuPDF
import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError as exc:
    raise SystemExit(
        "paddleocr is not installed. Install it with: pip install paddleocr"
    ) from exc


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int, max_pages: Optional[int] = None) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths: List[str] = []

    try:
        for i, page in enumerate(doc):
            if max_pages is not None and i >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi)
            image_path = os.path.join(output_dir, f"page_{i:04d}.png")
            pix.save(image_path)
            image_paths.append(image_path)
    finally:
        doc.close()

    return image_paths


def ocr_image(ocr: PaddleOCR, image_path: str) -> str:
    try:
        result = ocr.predict(image_path)
    except Exception:
        pix = fitz.Pixmap(image_path)
        if pix.alpha:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        result = ocr.predict(img)

    lines = []
    for item in result or []:
        if hasattr(item, "__getitem__"):
            try:
                rec_texts = item.get("rec_texts", None) if hasattr(item, "get") else item["rec_texts"]
                if rec_texts:
                    lines.extend([t for t in rec_texts if t])
                    continue
            except Exception:
                pass

        if isinstance(item, list):
            for block in item:
                if isinstance(block, list) and len(block) > 1 and isinstance(block[1], (list, tuple)):
                    text = block[1][0] if len(block[1]) > 0 else ""
                    if text:
                        lines.append(text)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from PDF with PaddleOCR.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_pdf", type=str, help="Path to input PDF file")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI for PDF pages (default: 150)")
    parser.add_argument("--lang", type=str, default="fr", help="PaddleOCR language code")
    parser.add_argument("--max-pages", type=int, default=None, help="Only process first N pages")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--textline-orientation", action="store_true", help="Enable orientation model (slower)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: reduce DPI to 120 and disable orientation")
    args = parser.parse_args()

    if not os.path.exists(args.input_pdf):
        raise SystemExit(f"Input PDF not found: {args.input_pdf}")

    base, _ = os.path.splitext(args.input_pdf)
    out_dir = f"{base}_images"
    merged_out = f"{base}.paddleocr.txt"

    dpi = args.dpi
    use_orientation = args.textline_orientation
    if args.fast:
        dpi = min(dpi, 120)
        use_orientation = False

    print(f"Rendering PDF pages to images: {out_dir} (dpi={dpi})")
    image_paths = pdf_to_images(args.input_pdf, out_dir, dpi, max_pages=args.max_pages)
    print(f"Total pages: {len(image_paths)}")

    print(f"Loading PaddleOCR model (lang={args.lang}, orientation={use_orientation})...")
    try:
        ocr_kwargs = dict(
            lang=args.lang,
            use_textline_orientation=use_orientation,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        if args.fast:
            ocr_kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
        ocr = PaddleOCR(**ocr_kwargs)
    except ModuleNotFoundError as exc:
        if exc.name == "paddle":
            raise SystemExit(
                "Missing dependency: paddlepaddle.\n"
                "Install with: pip install paddlepaddle"
            ) from exc
        raise

    merged_pages = []
    for i, image_path in enumerate(image_paths):
        print(f"OCR page {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
        page_text = ocr_image(ocr, image_path)
        txt_path = image_path.replace(".png", ".paddleocr.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(page_text)
        merged_pages.append(f"<PAGE:{i + 1}>\n{page_text}")

    with open(merged_out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(merged_pages))

    print(f"Done. Per-page text files in: {out_dir}")
    print(f"Merged output: {merged_out}")


if __name__ == "__main__":
    main()