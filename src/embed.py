import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("index")


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def build_index(chunks: list[dict], model: SentenceTransformer | None = None):
    """Embed every chunk and store the vectors in a FAISS inner-product index."""
    model = model or load_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product == cosine on normalized vectors
    index.add(embeddings)
    return index


def save_index(index, chunks: list[dict], index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(exist_ok=True)
    faiss.write_index(index, str(index_dir / "chunks.faiss"))
    (index_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


def load_index(index_dir: Path = INDEX_DIR):
    index = faiss.read_index(str(index_dir / "chunks.faiss"))
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks