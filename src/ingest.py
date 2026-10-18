import re
from collections import Counter
from pypdf import PdfReader


def remove_repeated_lines(raw_pages: list[str], min_fraction: float = 0.5) -> list[str]:
    """Drop running headers/footers: lines that repeat on many pages."""
    counts = Counter()
    for raw in raw_pages:
        for line in {l.strip() for l in raw.split("\n") if l.strip()}:
            counts[line] += 1
    threshold = max(3, int(len(raw_pages) * min_fraction))
    repeated = {line for line, c in counts.items() if c >= threshold}

    cleaned = []
    for raw in raw_pages:
        kept = [
            l for l in raw.split("\n")
            if l.strip() not in repeated and not l.strip().isdigit()  # also drop bare page numbers
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def clean_text(text: str) -> str:
    """Basic NLP text cleaning for PDF-extracted text."""
    text = re.sub(r"-\n(\w)", r"\1", text)      # join words hyphenated across lines
    text = re.sub(r"\s*\n\s*", " ", text)       # newlines -> spaces
    text = re.sub(r"\s{2,}", " ", text)         # collapse repeated whitespace
    # Small-caps extraction quirk in this PDF: unify the name's variants
    text = re.sub(r"(?i)concept\s*a\s*ttention", "ConceptAttention", text)
    return text.strip()


def extract_pages(pdf_path: str) -> list[dict]:
    """Return a list of {'page': int, 'text': str}, one entry per non-empty page."""
    reader = PdfReader(pdf_path)
    raw_pages = [page.extract_text() or "" for page in reader.pages]
    raw_pages = remove_repeated_lines(raw_pages)

    pages = []
    for i, raw in enumerate(raw_pages, start=1):
        text = clean_text(raw)
        if text:
            pages.append({"page": i, "text": text})
    return pages

def chunk_pages(
    pages: list[dict],
    chunk_size: int = 200,
    overlap: int = 40,
    min_page_words: int = 100,
) -> list[dict]:
    """
    Split each page into overlapping word-based chunks.
    Pages with fewer than `min_page_words` words (figure-only pages) are skipped.
    Returns a list of {'chunk_id': int, 'page': int, 'text': str}.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0
    for page in pages:
        words = page["text"].split()
        if len(words) < min_page_words:   # figure labels / near-empty page
            continue
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": " ".join(words[start:end]),
                }
            )
            chunk_id += 1
            if end >= len(words):
                break
            start = end - overlap
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