import json
import sys

import pandas as pd

from src.ingest import chunk_pages, extract_pages
from src.retrieve import DenseRetriever, HybridRetriever, TfidfRetriever

K = 5


def first_hit_rank(hits: list[dict], gold_pages: set[int]):
    """Rank (1-based) of the first retrieved chunk on a gold page, else None."""
    for rank, hit in enumerate(hits, start=1):
        if hit["page"] in gold_pages:
            return rank
    return None


def evaluate(retriever, questions: list[dict], k: int = K) -> dict:
    ranks = [
        first_hit_rank(retriever.search(q["question"], k=k), set(q["pages"]))
        for q in questions
    ]
    n = len(ranks)
    return {
        "Hit@1": sum(1 for r in ranks if r == 1) / n,
        "Hit@3": sum(1 for r in ranks if r is not None and r <= 3) / n,
        "Hit@5": sum(1 for r in ranks if r is not None) / n,
        "MRR": sum(1 / r for r in ranks if r) / n,
    }


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    with open("data/eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    pages = extract_pages(pdf_path)
    rows = []
    for chunk_size in (100, 150, 200):
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=int(chunk_size * 0.2))
        tfidf, dense = TfidfRetriever(chunks), DenseRetriever(chunks)
        hybrid = HybridRetriever(chunks, tfidf=tfidf, dense=dense)
        for name, retriever in [("TF-IDF", tfidf), ("Dense", dense), ("Hybrid", hybrid)]:
            rows.append(
                {"chunk_size": chunk_size, "retriever": name, "chunks": len(chunks),
                 **evaluate(retriever, questions)}
            )

    df = pd.DataFrame(rows).round(2)
    print(df.to_string(index=False))
    df.to_csv("results.csv", index=False)