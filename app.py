"""
app.py: Streamlit UI for the digital clone of Abdullah Bin Abid (Hybrid RAG).

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (see README.md)

The retrieval and generation logic lives in rag_core.py so the app, the tests
and the evaluation script all behave identically.
"""

import re

import pandas as pd
import streamlit as st

import rag_core as rc

st.set_page_config(page_title="Abdullah Bin Abid | AI Clone", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------------------
# Secrets (never hard-coded; read from Streamlit secrets)
# ---------------------------------------------------------------------------
def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:  # no secrets.toml at all
        return ""


API_KEY = get_secret("GROQ_API_KEY")
MODELS = rc.model_list(get_secret("GROQ_MODEL"))


# ---------------------------------------------------------------------------
# Load model and index ONCE per server process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_retriever():
    """Loads the embedding model, then loads embeddings.npy + chunks.json.
    If they are missing or out of date, it builds them automatically (first launch)."""
    status = {"model_error": None, "built_now": False, "save_error": None}
    try:
        model = rc.load_embedder()
    except Exception as e:  # e.g. no internet to download the model
        model = None
        status["model_error"] = f"{type(e).__name__}: {e}"

    saved = rc.load_saved_index()
    if saved is not None:
        chunks, emb = saved
    elif model is not None:
        try:
            chunks, emb = rc.build_and_save_index(model, verbose=False)
        except OSError as e:  # read-only disk: keep the index in memory instead
            status["save_error"] = str(e)
            chunks = rc.build_chunks()
            emb = rc.encode(model, [rc.embed_text_for(c) for c in chunks])
        status["built_now"] = True
    else:
        chunks, emb = rc.build_chunks(), None

    return rc.Retriever(chunks, emb, model), status


with st.spinner("Loading my notes and the embedding model (first launch can take about a minute)..."):
    retriever, load_status = load_retriever()


# ---------------------------------------------------------------------------
# Header and banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="background:#b91c1c;color:white;padding:14px 18px;border-radius:10px;
                font-size:1.15rem;font-weight:700;text-align:center;margin-bottom:12px;">
        ⚠️ This is an AI clone, not the real person.
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("🤖 Chat with Abdullah Bin Abid's Digital Clone")
st.caption(
    "Ask me about my studies, projects, skills, internship, hobbies or opinions. "
    "I answer only from documents Abdullah wrote about himself, using BM25, dense or hybrid retrieval."
)

# ---------------------------------------------------------------------------
# Sidebar: settings and status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=3, max_value=5, value=3)
    compare = st.checkbox("Also show the top-3 for every mode (comparison)", value=True)

    st.header("📊 Status")
    st.write(f"Chunks indexed: **{len(retriever.chunks)}** from **{len({c['source'] for c in retriever.chunks})}** files")
    if retriever.dense_available:
        st.success("Dense retrieval ready (all-MiniLM-L6-v2)")
        if load_status["built_now"]:
            st.info("Embeddings were built on this launch and cached.")
    else:
        st.error("Dense model could not load, so only BM25 works right now.")
        with st.expander("Details"):
            st.code(load_status["model_error"] or "unknown")
    if API_KEY:
        st.success(f"AI answers on (Groq, {MODELS[0]})")
    else:
        st.warning("No GROQ_API_KEY in secrets: showing templated answers from my notes.")

    st.header("🧪 Evaluation")
    run_eval = st.button("Run evaluation report", help="Top-3 chunks per mode for the report questions, plus hit@3")


# ---------------------------------------------------------------------------
# Question form
# ---------------------------------------------------------------------------
mode = st.radio("Retrieval mode", rc.MODES, index=2, horizontal=True)
if mode != "BM25 only" and not retriever.dense_available:
    st.warning("Dense model unavailable, so I'm falling back to BM25 for this question.")
    mode = "BM25 only"

with st.form("ask", clear_on_submit=False):
    question = st.text_input("Your question", placeholder="e.g. What did you work on at WPBrigade?")
    submitted = st.form_submit_button("Ask", type="primary")


def hits_table(hits):
    rows = []
    for h in hits:
        rows.append(
            {
                "Rank": h.rank,
                "Source file": h.chunk["source"],
                "Chunk ID": h.chunk["chunk_id"],
                "Score": round(h.score, 4),
                "BM25 rank": str(h.bm25_rank) if h.bm25_rank is not None else "-",
                "Dense rank": str(h.dense_rank) if h.dense_rank is not None else "-",
                "Preview": " ".join(h.chunk["text"].split()[:18]) + "...",
            }
        )
    return pd.DataFrame(rows)


if submitted and question.strip():
    with st.spinner("Thinking..."):
        result = rc.answer_question(question.strip(), retriever, mode, top_k, API_KEY, MODELS)

    st.subheader("💬 Answer")
    if result["error"]:
        st.warning(f"AI answer unavailable: {result['error']}\n\nShowing a templated answer from the retrieved chunks instead.")
    st.markdown(result["answer"])
    meta = f"Mode: **{mode}**"
    if result["route"] != "normal":
        meta += f" · Router: **{result['route']}** question (answered by a fixed safety rule)"
    elif result["model"]:
        meta += f" · Model: `{result['model']}`"
    st.caption(meta)

    st.subheader(f"📚 Retrieved chunks ({mode})")
    score_note = {
        "BM25 only": "Score = BM25 relevance (higher is better).",
        "Dense only": "Score = cosine similarity between question and chunk embeddings (-1 to 1).",
        "Hybrid": "Score = Reciprocal Rank Fusion: 1/(60 + BM25 rank) + 1/(60 + dense rank).",
    }[mode]
    st.caption(score_note)
    st.dataframe(hits_table(result["hits"]), hide_index=True, use_container_width=True)
    for h in result["hits"]:
        with st.expander(f"#{h.rank} · {h.chunk['source']} · {h.chunk['chunk_id']} · score {h.score:.4f}"):
            st.markdown(re.sub(r"(?m)^#+\s*(.*)$", r"**\1**", h.chunk["text"]))

    if compare and retriever.dense_available:
        st.subheader("🔍 Mode comparison (top-3 each)")
        cols = st.columns(3)
        for col, m in zip(cols, rc.MODES):
            with col:
                st.markdown(f"**{m}**")
                for h in retriever.search(question.strip(), m, 3):
                    st.markdown(f"{h.rank}. `{h.chunk['chunk_id']}` ({h.score:.4f})")
elif submitted:
    st.info("Type a question first.")


# ---------------------------------------------------------------------------
# Evaluation report (copy this output into the README report)
# ---------------------------------------------------------------------------
if run_eval:
    if not retriever.dense_available:
        st.error("The evaluation needs the dense model, which is not loaded.")
    else:
        import eval_report

        st.subheader("🧪 Evaluation report")
        st.caption("Copy everything in the box below and paste it back to update the README report.")
        st.code(eval_report.run_report(retriever), language="text")

st.divider()
st.caption(
    "Built for Assignment 2 (Hybrid RAG). Retrieval: rank_bm25 + all-MiniLM-L6-v2 with numpy cosine "
    "similarity + Reciprocal Rank Fusion. Generation: Groq free tier. This is an AI clone, not the real person."
)
