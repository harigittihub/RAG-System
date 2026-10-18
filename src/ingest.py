import re
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Basic NLP text cleaning for PDF-extracted text."""
    text = re.sub(r"-\n(\w)", r"\1", text)   # join words hyphenated across lines
    text = re.sub(r"\s*\n\s*", " ", text)    # newlines -> spaces
    text = re.sub(r"\s{2,}", " ", text)      # collapse repeated whitespace
    return text.strip()


def extract_pages(pdf_path: str) -> list[dict]:
    """Return a list of {'page': int, 'text': str}, one entry per non-empty page."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if text:  # skip blank pages
            pages.append({"page": i, "text": text})
    return pages

def chunk_pages(
    pages: list[dict], chunk_size: int = 200, overlap: int = 40
) -> list[dict]:
    """
    Split each page into overlapping word-based chunks.
    Returns a list of {'chunk_id': int, 'page': int, 'text': str}.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": " ".join(chunk_words),
                }
            )
            chunk_id += 1
            if end >= len(words):
                break
            start = end - overlap  # step back by `overlap` words
    return chunks

if __name__ == "__main__":
    import sys

    pages = extract_pages(sys.argv[1])
    chunks = chunk_pages(pages, chunk_size=200, overlap=40)

    lengths = [len(c["text"].split()) for c in chunks]
    print(f"Pages: {len(pages)} | Chunks: {len(chunks)}")
    print(f"Chunk length (words): min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths) / len(lengths):.0f}")

    print("\n--- Chunk 0 ---")
    print(chunks[0]["text"][:300], "...")
    print("\n--- Chunk 1 (note the overlap with chunk 0) ---")
    print(f"[page {chunks[1]['page']}]", chunks[1]["text"][:300], "...")