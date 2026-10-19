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

class DenseRetriever:
    """Semantic retriever: sentence embeddings + FAISS cosine search."""

    def __init__(self, chunks: list[dict], model=None, index=None):
        from src.embed import build_index, load_model

        self.chunks = chunks
        self.model = model if model is not None else load_model()
        self.index = index if index is not None else build_index(chunks, self.model)

    def search(self, query: str, k: int = 3) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(query_vec, k)
        return [
            {**self.chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]

class HybridRetriever:
    """Reciprocal Rank Fusion of TF-IDF (lexical) and dense (semantic) rankings."""

    def __init__(self, chunks: list[dict], rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.tfidf = TfidfRetriever(chunks)
        self.dense = DenseRetriever(chunks)

    def search(self, query: str, k: int = 4, candidates: int = 20) -> list[dict]:
        fused: dict[int, float] = {}
        for retriever in (self.tfidf, self.dense):
            for rank, hit in enumerate(retriever.search(query, k=candidates), start=1):
                fused[hit["chunk_id"]] = fused.get(hit["chunk_id"], 0.0) + 1.0 / (self.rrf_k + rank)
        top_ids = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**self.by_id[cid], "score": fused[cid]} for cid in top_ids]

if __name__ == "__main__":
    import sys

    from src.ingest import chunk_pages, extract_pages

    pdf_path, query = sys.argv[1], sys.argv[2]
    chunks = chunk_pages(extract_pages(pdf_path), chunk_size=150, overlap=30)

    retrievers = [("TF-IDF", TfidfRetriever(chunks)), ("Dense", DenseRetriever(chunks))]
    for name, retriever in retrievers:
        print(f"\n===== {name} =====")
        for rank, hit in enumerate(retriever.search(query, k=3), start=1):
            print(f"#{rank} | page {hit['page']} | score {hit['score']:.3f}")
            print(hit["text"][:250], "...\n")