"""
rag_core.py
Shared retrieval and generation logic for the Abdullah Bin Abid digital clone.

Used by app.py (Streamlit UI), build_index.py, eval_report.py and test_clone.py,
so every script retrieves and answers in exactly the same way.

Pipeline: load docs -> chunk (200-300 words, overlap) -> BM25 + dense embeddings
-> retrieve (BM25 / Dense / Hybrid with Reciprocal Rank Fusion) -> LLM answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHUNKS_PATH = ROOT / "chunks.json"
EMB_PATH = ROOT / "embeddings.npy"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking: aim for ~230 words, never go past 300, overlap ~40 words.
# all-MiniLM-L6-v2 reads at most 256 word-pieces (~190-200 English words), so
# we keep chunks near the low end of the 200-300 range to lose as little text
# as possible to truncation.
TARGET_WORDS = 230
MAX_WORDS = 300
OVERLAP_WORDS = 40

RRF_K = 60  # Reciprocal Rank Fusion constant: score = sum(1 / (RRF_K + rank))

MODES = ["BM25 only", "Dense only", "Hybrid"]

OWNER_NAME = "Abdullah Bin Abid"
NOT_IN_NOTES = "I don't have that in my notes."

# Groq free tier (OpenAI-compatible). Llama models left the free plan in Aug 2026,
# so the default is gpt-oss-20b with gpt-oss-120b as a fallback.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
REQUEST_TIMEOUT = 20  # seconds

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
STOPWORDS = set(
    """a an the and or but if of to in on at for from by with about as into is are was
    were be been being am do does did doing have has had having i me my mine myself you
    your yours we our us he she it its they them their this that these those what which
    who whom whose when where why how there here so than too very can could would should
    will just not no nor only own same such some any each few more most other all both
    up down out off over under again further then once s t don ve ll re d m""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, keep letters/digits (and dots inside tokens like 'chess.js'), drop stopwords."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Loading and chunking
# ---------------------------------------------------------------------------
def load_documents(data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    """Return [(file_name, text)] for every .md / .txt file in data/, sorted by name."""
    docs = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    if not docs:
        raise FileNotFoundError(f"No .md or .txt files found in {data_dir}")
    return docs


def _split_units(text: str) -> list[str]:
    """Split a document into paragraphs. Very long paragraphs are split into sentences.
    Keeping paragraphs whole means an FAQ question stays with its answer."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if word_count(para) <= MAX_WORDS:
            units.append(para)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
    return units


def _overlap_tail(units: list[str], n_words: int) -> list[str]:
    """Take whole units from the end of a chunk until we have ~n_words of overlap."""
    tail, total = [], 0
    for u in reversed(units):
        if total >= n_words:
            break
        tail.insert(0, u)
        total += word_count(u)
    # Never carry over the whole chunk as overlap.
    if len(tail) == len(units):
        tail = tail[1:]
    return tail


def chunk_document(file_name: str, text: str) -> list[dict]:
    """Greedy paragraph packing: fill a chunk to ~TARGET_WORDS (max MAX_WORDS), then start
    the next chunk with the last ~OVERLAP_WORDS of the previous one."""
    units = _split_units(text)
    title = units[0].lstrip("# ").strip() if units and units[0].startswith("#") else file_name

    chunks_units: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        w = word_count(unit)
        if current and (current_words + w > MAX_WORDS or current_words >= TARGET_WORDS):
            chunks_units.append(current)
            current = _overlap_tail(current, OVERLAP_WORDS)
            current_words = sum(word_count(u) for u in current)
        current.append(unit)
        current_words += w
    if current:
        # Merge a tiny final chunk into the previous one if it still fits.
        if chunks_units and current_words < 80:
            prev = chunks_units[-1]
            merged = prev + [u for u in current if u not in prev]
            if sum(word_count(u) for u in merged) <= MAX_WORDS:
                chunks_units[-1] = merged
                current = []
        if current:
            chunks_units.append(current)

    stem = Path(file_name).stem
    out = []
    for i, cu in enumerate(chunks_units):
        body = "\n\n".join(cu)
        out.append(
            {
                "chunk_id": f"{stem}-{i:02d}",
                "source": file_name,
                "title": title,
                "text": body,
                "n_words": word_count(body),
            }
        )
    return out


def build_chunks(data_dir: Path = DATA_DIR) -> list[dict]:
    chunks = []
    for name, text in load_documents(data_dir):
        chunks.extend(chunk_document(name, text))
    return chunks


def data_fingerprint(data_dir: Path = DATA_DIR) -> str:
    """Hash of all document contents plus chunk settings, so the index rebuilds
    automatically whenever a document or the chunking changes."""
    h = hashlib.sha256()
    h.update(f"{TARGET_WORDS}-{MAX_WORDS}-{OVERLAP_WORDS}-{EMBED_MODEL_NAME}".encode())
    for name, text in load_documents(data_dir):
        h.update(name.encode())
        h.update(text.encode())
    return h.hexdigest()[:16]


def embed_text_for(chunk: dict) -> str:
    """Text that gets embedded: the document title gives each chunk extra context."""
    return f"{chunk['title']}\n{chunk['text']}"


# ---------------------------------------------------------------------------
# Embedding model and index
# ---------------------------------------------------------------------------
def load_embedder():
    """Load all-MiniLM-L6-v2 on CPU (downloads ~90 MB the first time)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME, device="cpu")


def encode(model, texts: list[str]) -> np.ndarray:
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    vecs = vecs.astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)  # unit length, so dot product = cosine


def build_and_save_index(model=None, verbose: bool = True) -> tuple[list[dict], np.ndarray]:
    """Chunk all documents, embed them once, save chunks.json and embeddings.npy."""
    chunks = build_chunks()
    if model is None:
        model = load_embedder()
    t0 = time.time()
    emb = encode(model, [embed_text_for(c) for c in chunks])
    np.save(EMB_PATH, emb)
    payload = {"fingerprint": data_fingerprint(), "model": EMBED_MODEL_NAME, "chunks": chunks}
    CHUNKS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"Embedded {len(chunks)} chunks in {time.time() - t0:.1f}s -> {EMB_PATH.name}, {CHUNKS_PATH.name}")
    return chunks, emb


def load_saved_index() -> tuple[list[dict], np.ndarray] | None:
    """Return the saved index if it exists and matches the current documents."""
    if not (CHUNKS_PATH.exists() and EMB_PATH.exists()):
        return None
    try:
        payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        emb = np.load(EMB_PATH)
    except Exception:
        return None
    if payload.get("fingerprint") != data_fingerprint():
        return None
    if emb.shape[0] != len(payload["chunks"]):
        return None
    return payload["chunks"], emb


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    chunk: dict
    score: float
    rank: int
    bm25_rank: int | None = None
    dense_rank: int | None = None


class Retriever:
    """Holds the BM25 index, the dense embeddings and the embedding model."""

    def __init__(self, chunks: list[dict], embeddings: np.ndarray | None, model=None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model
        self.bm25 = BM25Okapi([tokenize(c["title"] + " " + c["text"]) for c in chunks])

    @property
    def dense_available(self) -> bool:
        return self.embeddings is not None and self.model is not None

    # Full rankings over all chunks (index lists, best first) -------------
    def bm25_scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        if not q:
            return np.zeros(len(self.chunks))
        return np.asarray(self.bm25.get_scores(q), dtype=float)

    def dense_scores(self, query: str) -> np.ndarray:
        if not self.dense_available:
            raise RuntimeError("Dense retrieval is unavailable (embedding model not loaded).")
        q = encode(self.model, [query])[0]
        return self.embeddings @ q  # cosine similarity, because both sides are unit length

    @staticmethod
    def _order(scores: np.ndarray) -> list[int]:
        # Stable sort: ties keep document order, so results are deterministic.
        return list(np.argsort(-scores, kind="stable"))

    # Public API ------------------------------------------------------------
    def search(self, query: str, mode: str, k: int = 3) -> list[Hit]:
        if mode == "BM25 only":
            s = self.bm25_scores(query)
            order = self._order(s)
            return [Hit(self.chunks[i], float(s[i]), r + 1, bm25_rank=r + 1) for r, i in enumerate(order[:k])]

        if mode == "Dense only":
            s = self.dense_scores(query)
            order = self._order(s)
            return [Hit(self.chunks[i], float(s[i]), r + 1, dense_rank=r + 1) for r, i in enumerate(order[:k])]

        if mode == "Hybrid":
            return self.hybrid(query, k)

        raise ValueError(f"Unknown mode: {mode}")

    def hybrid(self, query: str, k: int = 3) -> list[Hit]:
        """Reciprocal Rank Fusion over the FULL BM25 and dense rankings.
        score(chunk) = 1/(60 + bm25_rank) + 1/(60 + dense_rank), ranks start at 1.
        Chunks with zero BM25 score (no shared words) get no BM25 contribution."""
        bm = self.bm25_scores(query)
        de = self.dense_scores(query)
        bm_order = self._order(bm)
        de_order = self._order(de)
        bm_rank = {i: r + 1 for r, i in enumerate(bm_order) if bm[i] > 0}
        de_rank = {i: r + 1 for r, i in enumerate(de_order)}

        fused = np.zeros(len(self.chunks))
        for i in range(len(self.chunks)):
            if i in bm_rank:
                fused[i] += 1.0 / (RRF_K + bm_rank[i])
            fused[i] += 1.0 / (RRF_K + de_rank[i])

        order = self._order(fused)
        return [
            Hit(self.chunks[i], float(fused[i]), r + 1, bm25_rank=bm_rank.get(i), dense_rank=de_rank[i])
            for r, i in enumerate(order[:k])
        ]


# ---------------------------------------------------------------------------
# Question routing (simple agentic step before retrieval)
# ---------------------------------------------------------------------------
PRIVATE_PATTERNS = [
    r"\bphone\b", r"\bmobile (number|no)\b", r"\bcell\b", r"\bwhats\s?app\b", r"\bcontact number\b",
    r"\b(home|house|street|postal|mailing) ?address\b", r"\bwhere (do|does) (you|he) live\b",
    r"\bwhere is your (house|home)\b", r"\baddress\b", r"\bcnic\b", r"\bpassport\b",
    r"\bid (card|number)\b", r"\bnational id\b", r"\bpassword\b", r"\bbank\b", r"\baccount number\b",
    r"\bcredit card\b", r"\biban\b", r"\bsocial security\b", r"\broll (number|no)\b", r"\bemail address\b",
]
IDENTITY_PATTERNS = [
    r"\bare you (the )?real\b", r"\bare you (a )?(human|person|bot|robot|an? ai)\b",
    r"\bis this (really|actually) (you|abdullah)\b", r"\bam i (talking|chatting|speaking) (to|with) (the )?real\b",
    r"\bare you really abdullah\b", r"\bare you actually abdullah\b", r"\bwho (made|built|created) you\b",
]

PRIVATE_REPLY = (
    "Sorry, I can't share that. I don't give out private details like phone numbers, "
    "addresses, ID numbers or account details. Happy to chat about my studies, projects or interests though!"
)
IDENTITY_REPLY = (
    f"No, I'm not the real person. I'm an AI clone of {OWNER_NAME}, built from documents he wrote "
    "about himself. I can only answer from those notes."
)


def route_question(question: str) -> str:
    """Return 'private', 'identity' or 'normal'."""
    q = question.lower()
    if any(re.search(p, q) for p in PRIVATE_PATTERNS):
        return "private"
    if any(re.search(p, q) for p in IDENTITY_PATTERNS):
        return "identity"
    return "normal"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
PERSONA_PROMPT = """You are the AI clone of Abdullah Bin Abid, a 7th semester (senior year) Computer Science student at the University of Management and Technology (UMT) who is particularly interested in Artificial Intelligence and Machine Learning.
Speak in the first person, as Abdullah. Your tone is casual but professional, and you adapt to the question: relaxed for casual questions, more formal for serious ones. Keep answers short (1 to 4 sentences). You sometimes use an emoji and you love a good idiom, but never overdo either.

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say exactly: "I don't have that in my notes." Do not guess or invent facts about me. Do not add details that are not in the context.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, say you are an AI clone of Abdullah Bin Abid.
5. Do not mention "the context" or "the documents" in your answer; just answer naturally as me."""

USER_TEMPLATE = """CONTEXT:
{retrieved_chunks}

QUESTION:
{question}"""


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.chunk['source']} | {h.chunk['chunk_id']}]\n{h.chunk['text']}" for h in hits)


class LLMError(Exception):
    """Raised with a friendly, user-facing message."""


def call_groq(question: str, hits: list[Hit], api_key: str, models: list[str] | None = None,
              timeout: int = REQUEST_TIMEOUT) -> tuple[str, str]:
    """Return (answer, model_used). Raises LLMError with a friendly message on failure."""
    if not api_key:
        raise LLMError("No API key found. Add GROQ_API_KEY to the app's secrets to enable AI answers.")
    messages = [
        {"role": "system", "content": PERSONA_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(retrieved_chunks=format_context(hits), question=question)},
    ]
    last_error = "Unknown error."
    for model in models or DEFAULT_MODELS:
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_completion_tokens": 700,
        }
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"  # keeps answers fast and inside free-tier token limits
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            raise LLMError(f"The AI service took longer than {timeout} seconds to reply. Please try again.")
        except requests.exceptions.RequestException:
            raise LLMError("Couldn't reach the AI service (network error). Please try again in a moment.")

        if resp.status_code == 200:
            try:
                msg = resp.json()["choices"][0]["message"]
                answer = (msg.get("content") or "").strip()
            except (KeyError, IndexError, ValueError):
                answer = ""
            if answer:
                return answer, model
            last_error = "The AI service returned an empty answer."
            continue
        if resp.status_code == 429:
            wait = resp.headers.get("retry-after")
            hint = f" Try again in about {wait} seconds." if wait else " Please wait a minute and try again."
            raise LLMError("The free AI quota is busy right now (rate limit reached)." + hint)
        if resp.status_code in (401, 403):
            raise LLMError("The API key was rejected. Check that GROQ_API_KEY in the secrets is correct.")
        if resp.status_code in (400, 404):
            # Usually a model that was renamed or removed: try the next model in the list.
            last_error = f"Model '{model}' is not available ({resp.status_code})."
            continue
        if resp.status_code >= 500:
            raise LLMError("The AI service is having problems right now. Please try again shortly.")
        last_error = f"Unexpected response from the AI service ({resp.status_code})."
    raise LLMError(last_error + " Set GROQ_MODEL in secrets to a model listed at console.groq.com/docs/models.")


def templated_answer(hits: list[Hit]) -> str:
    """Fallback when the LLM is unavailable: show the most relevant passage."""
    if not hits:
        return NOT_IN_NOTES
    top = hits[0].chunk
    excerpt = " ".join(top["text"].split()[:70])
    return (
        f"(AI generation is unavailable, so here is the most relevant passage from my notes, "
        f"from {top['source']}.)\n\n{excerpt}..."
    )


def answer_question(question: str, retriever: Retriever, mode: str, k: int, api_key: str,
                    models: list[str] | None = None) -> dict:
    """Full pipeline used by the app and the tests.
    Returns dict(answer, hits, route, used_llm, model, error)."""
    route = route_question(question)
    hits = retriever.search(question, mode, k)
    if route == "private":
        return dict(answer=PRIVATE_REPLY, hits=hits, route=route, used_llm=False, model=None, error=None)
    if route == "identity":
        return dict(answer=IDENTITY_REPLY, hits=hits, route=route, used_llm=False, model=None, error=None)
    try:
        answer, model = call_groq(question, hits, api_key, models)
        return dict(answer=answer, hits=hits, route=route, used_llm=True, model=model, error=None)
    except LLMError as e:
        return dict(answer=templated_answer(hits), hits=hits, route=route, used_llm=False, model=None, error=str(e))


# ---------------------------------------------------------------------------
# Secrets helper for local scripts (app.py uses st.secrets directly)
# ---------------------------------------------------------------------------
def read_local_secret(name: str) -> str:
    """Read a key from the environment or .streamlit/secrets.toml (for test/eval scripts)."""
    import os

    if os.environ.get(name):
        return os.environ[name]
    path = ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # Python 3.10
            import toml as tomllib  # streamlit installs 'toml'
            return str(tomllib.load(str(path)).get(name, ""))
        with open(path, "rb") as f:
            return str(tomllib.load(f).get(name, ""))
    return ""


def model_list(preferred: str = "") -> list[str]:
    preferred = (preferred or "").strip()
    return ([preferred] if preferred else []) + [m for m in DEFAULT_MODELS if m != preferred]
