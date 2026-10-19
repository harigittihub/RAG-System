import ollama
import re

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about a document. "
    "Use the provided context passages to answer; each passage is labeled with its "
    "page, like [p.4]. Answer in 2-4 sentences and cite the pages you used, e.g. [p.4]. "
    "Say \"I couldn't find that in the document.\" only if none of the passages "
    "contain information relevant to the question."
)


def build_prompt(query: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[p.{h['page']}] {h['text']}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def generate_answer(query: str, hits: list[dict], model: str = MODEL) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, hits)},
        ],
        options={"temperature": 0, "num_ctx":4096},  # deterministic, less creative drift
    )
    return response["message"]["content"].strip()

def cited_pages(answer: str) -> set[int]:
    """Pages the model cites, e.g. [p.5] or [p.1-p.4] -> {5} / {1, 4}."""
    return {int(n) for n in re.findall(r"p\.(\d+)", answer)}


def unsupported_citations(answer: str, hits: list[dict]) -> list[int]:
    """Cited pages that were NOT in the retrieved context (likely hallucinated)."""
    return sorted(cited_pages(answer) - {h["page"] for h in hits})