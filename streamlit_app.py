import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.embed import load_model
from src.generate import MODEL, cited_pages, generate_answer, unsupported_citations
from src.ingest import chunk_pages, extract_pages
from src.retrieve import DenseRetriever, HybridRetriever, TfidfRetriever

st.set_page_config(page_title="PDF RAG", layout="wide")


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Indexing document...")
def build_retrievers(file_bytes: bytes, chunk_size: int):
    pages = extract_pages(io.BytesIO(file_bytes))
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=int(chunk_size * 0.2))
    tfidf = TfidfRetriever(chunks)
    dense = DenseRetriever(chunks, model=get_model())
    hybrid = HybridRetriever(chunks, tfidf=tfidf, dense=dense)
    retrievers = {"TF-IDF": tfidf, "Dense": dense, "Hybrid": hybrid}
    return retrievers, len(pages), len(chunks)


# ---------- Sidebar ----------
st.sidebar.header("Document")
uploaded = st.sidebar.file_uploader("Upload a PDF", type="pdf")
samples = sorted(Path("data").glob("*.pdf"))
if uploaded is not None:
    doc_name, doc_bytes = uploaded.name, uploaded.getvalue()
elif samples:
    choice = st.sidebar.selectbox(
        "...or use a PDF from data/", samples, format_func=lambda p: p.name
    )
    doc_name, doc_bytes = choice.name, choice.read_bytes()
else:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

st.sidebar.header("Retrieval")
retriever_name = st.sidebar.selectbox("Retriever", ["TF-IDF", "Dense", "Hybrid"], index=0)
chunk_size = st.sidebar.select_slider("Chunk size (words)", options=[100, 150, 200], value=100)
top_k = st.sidebar.slider("Chunks sent to the LLM (top-k)", 3, 8, 6)

st.sidebar.header("Generation")
use_llm = st.sidebar.checkbox("Generate answer with Ollama", value=True)
model_name = st.sidebar.text_input("Ollama model", value=MODEL)

# ---------- Main ----------
retrievers, n_pages, n_chunks = build_retrievers(doc_bytes, chunk_size)

st.title("Ask your PDF")
st.caption(f"{doc_name}  |  {n_pages} pages with text  |  {n_chunks} chunks")

tab_ask, tab_eval = st.tabs(["Ask", "Evaluation"])

with tab_ask:
    with st.form("ask_form"):
        question = st.text_input(
            "Your question", placeholder="e.g. Which datasets were used for evaluation?"
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        q = question.strip()
        hits = retrievers[retriever_name].search(q, k=top_k)
        answer, error = None, None
        if use_llm:
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(q, hits, model=model_name)
                except Exception as exc:  # Ollama not running, model not pulled, etc.
                    error = str(exc)
        st.session_state["last"] = {
            "question": q, "hits": hits, "answer": answer,
            "error": error, "retriever": retriever_name, "k": top_k,
        }

    last = st.session_state.get("last")
    if last:
        st.markdown(f"**Q:** {last['question']}")
        st.caption(f"Retriever: {last['retriever']}  |  top-k: {last['k']}")

        st.subheader("Answer")
        if last["error"]:
            st.error(
                "Could not get an answer from the LLM. Is Ollama running and the model "
                f"pulled (`ollama list`)? Details: {last['error']}"
            )
        elif last["answer"]:
            st.write(last["answer"])
            bad = unsupported_citations(last["answer"], last["hits"])
            if bad:
                st.warning(
                    f"The answer cites pages that were not in the retrieved context: {bad}"
                )
        else:
            st.caption("Generation is switched off, showing retrieved passages only.")

        cited = cited_pages(last["answer"]) if last["answer"] else set()
        st.subheader("Sources")
        for h in last["hits"]:
            tag = "  |  cited by the answer" if h["page"] in cited else ""
            with st.expander(f"Page {h['page']}  |  score {h['score']:.3f}{tag}"):
                st.write(h["text"])

with tab_eval:
    st.subheader("Retrieval evaluation")
    results_path = Path("results.csv")
    if results_path.exists():
        df = pd.read_csv(results_path)
        metric = st.selectbox("Metric", ["MRR", "Hit@1", "Hit@3", "Hit@5"])
        pivot = df.pivot(index="chunk_size", columns="retriever", values=metric)
        pivot.index = pivot.index.astype(str)
        st.bar_chart(pivot)
        st.dataframe(df, hide_index=True)
        st.caption("Page-level matching on a hand-labeled question set (data/eval_questions.json).")
    else:
        st.info("Run `python -m src.evaluate data/<your>.pdf` to generate results.csv.")