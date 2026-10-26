import sys

from src.generate import generate_answer, unsupported_citations
from src.ingest import chunk_pages, extract_pages
from src.retrieve import HybridRetriever, TfidfRetriever


def main(pdf_path: str) -> None:
    chunks = chunk_pages(extract_pages(pdf_path), chunk_size=100, overlap=20)
    retriever = TfidfRetriever(chunks)
    print(f"Loaded {len(chunks)} chunks. Ask a question (empty line or 'quit' to exit).")

    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in {"", "quit", "exit"}:
            break

        hits = retriever.search(query, k=6)
        answer = generate_answer(query, hits)
        print("\nAnswer:", answer)

        bad = unsupported_citations(answer, hits)
        if bad:
            print(f"\nWARNING: cited pages not in retrieved context: {bad}")

        print("\nSources:")
        for h in hits:
            snippet = h["text"][:110].replace("\n", " ")
            print(f"  p.{h['page']}  ({h['score']:.3f})  {snippet}...")


if __name__ == "__main__":
    main(sys.argv[1])