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


if __name__ == "__main__":
    import sys

    pages = extract_pages(sys.argv[1])
    print(f"Extracted {len(pages)} non-empty pages")
    print(f"--- Page {pages[0]['page']} preview ---")
    print(pages[0]["text"][:500])