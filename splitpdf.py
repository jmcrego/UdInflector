
from pypdf import PdfReader, PdfWriter
import os
import sys

def split_pdf_with_overlap(input_pdf, output_dir, chunk_size=5, overlap=1):
    """
    Splits a PDF into chunks of N pages with O-page overlap.

    :param input_pdf: Path to input PDF
    :param output_dir: Directory to save chunks
    :param chunk_size: Number of pages per chunk (N)
    :param overlap: Number of overlapping pages (O)
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)

    os.makedirs(output_dir, exist_ok=True)

    step = chunk_size - overlap
    chunk_index = 0

    for start in range(0, total_pages, step):
        end = min(start + chunk_size, total_pages)

        writer = PdfWriter()

        for i in range(start, end):
            writer.add_page(reader.pages[i])

        # output_path = os.path.join(output_dir, f"chunk_{chunk_index}.pdf")
        output_path = os.path.join(output_dir, f"chunk_{start}-{end-1}.pdf")

        with open(output_path, "wb") as f:
            writer.write(f)

        print(f"Saved: {output_path} (pages {start}–{end-1})")

        chunk_index += 1

        if end == total_pages:
            break


# Example usage
if __name__ == "__main__":
    input_pdf = sys.argv[1]  # Path to input PDF
    output_dir = input_pdf.replace(".pdf", "_chunks")  # Output directory based on input PDF name
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    split_pdf_with_overlap(
        input_pdf=input_pdf,  # Path to input PDF
        output_dir=output_dir,  # Output directory based on input PDF name
        chunk_size=5,   # N pages per chunk
        overlap=1       # O pages overlap
    )