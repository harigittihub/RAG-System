import sys

from src.generate import generate_answer
from src.ingest import chunk_pages, extract_pages
from src.retrieve import HybridRetriever


def main(pdf_path: str) -> None:
    chunks = chunk_pages(extract_pages(pdf_path), chunk_size=150, overlap=30)
    retriever = HybridRetriever(chunks)
    print(f"Loaded {len(chunks)} chunks. Ask a question (empty line or 'quit' to exit).")

    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in {"", "quit", "exit"}:
            break
        hits = retriever.search(query, k=5)
        print("\nAnswer:", generate_answer(query, hits))
        print("\nSources:")
        for h in hits:
            snippet = h["text"][:110].replace("\n", " ")
            print(f"  p.{h['page']}  ({h['score']:.3f})  {snippet}...")


if __name__ == "__main__":
    main(sys.argv[1])