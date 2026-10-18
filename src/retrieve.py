import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """Lexical retriever: TF-IDF vectors + cosine similarity."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams ("saliency maps")
            sublinear_tf=True,    # dampen very frequent words
        )
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        top_ids = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top_ids]


if __name__ == "__main__":
    import sys

    from src.ingest import chunk_pages, extract_pages

    pdf_path, query = sys.argv[1], sys.argv[2]
    chunks = chunk_pages(extract_pages(pdf_path), chunk_size=200, overlap=40)
    retriever = TfidfRetriever(chunks)

    for rank, hit in enumerate(retriever.search(query, k=3), start=1):
        print(f"\n#{rank} | page {hit['page']} | score {hit['score']:.3f}")
        print(hit["text"][:400], "...")