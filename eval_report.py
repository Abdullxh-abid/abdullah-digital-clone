"""
eval_report.py: prints the REAL retrieval results used in the README report.

For each of the 5 report questions it prints the top-3 chunks for BM25, Dense
and Hybrid (source file, chunk ID, score, and for hybrid the BM25/dense ranks),
then computes hit@3 for every mode on a small hand-labeled set.

Usage:  python eval_report.py
(The same report can be produced in the deployed app: sidebar > "Run evaluation report".)
"""

import re

import rag_core as rc

# The 5 report questions (Q1 = exact rare keyword, Q2 = paraphrase with no shared words).
REPORT_QUESTIONS = [
    ("Q1 (rare keyword)", "What did you do at WPBrigade?"),
    ("Q2 (paraphrase, no shared words)", "When are you most productive?"),
    ("Q3 (paraphrase)", "Which football club do you cheer for?"),
    ("Q4 (keyword + meaning)", "How accurate was your MobileNetV2 model on PlantVillage?"),
    ("Q5 (opinion)", "What is your view on AI being used for creative work?"),
]

# Hand-labeled set for hit@3: (question, regex for the answer text).
# A "hit" means at least one of the top-3 retrieved chunks actually CONTAINS the answer
# (stricter than "comes from the right file", since the FAQ touches almost every topic).
LABELED_SET = [
    ("What did you do at WPBrigade?", r"AI models for foreign companies"),
    ("When are you most productive?", r"night owl"),
    ("Which football club do you cheer for?", r"Barcelona"),
    ("How accurate was your MobileNetV2 model?", r"94\.45"),
    ("Can your software tell if a crop is sick from a photo?", r"(leaf|leaves|crop disease)"),
    ("Can diners see their meal in 3D before ordering?", r"life-size"),
    ("Which country do you want to study in after graduating?", r"Canada"),
    ("What features help colour-blind users in your games?", r"[Cc]olor-blind"),
    ("What do you eat when you want comfort food?", r"[Bb]iryani"),
    ("What tip would you give a first-year student?", r"2 to 3 projects"),
]


def hit_at_k(retriever: rc.Retriever, mode: str, k: int = 3) -> tuple[float, list[bool]]:
    hits = []
    for q, answer_pattern in LABELED_SET:
        top = retriever.search(q, mode, k)
        hits.append(any(re.search(answer_pattern, h.chunk["text"]) for h in top))
    return sum(hits) / len(hits), hits


def run_report(retriever: rc.Retriever) -> str:
    lines = ["=== RETRIEVAL REPORT (top-3 per mode) ==="]
    for label, q in REPORT_QUESTIONS:
        lines.append("")
        lines.append(f"{label}: {q}")
        lines.append(f"  Query tokens used by BM25: {rc.tokenize(q)}")
        for mode in rc.MODES:
            lines.append(f"  [{mode}]")
            for h in retriever.search(q, mode, 3):
                extra = ""
                if mode == "Hybrid":
                    extra = f"  (bm25_rank={h.bm25_rank or '-'}, dense_rank={h.dense_rank})"
                preview = " ".join(h.chunk["text"].split()[:12])
                lines.append(f"    {h.rank}. {h.chunk['chunk_id']:<30} score={h.score:.4f}{extra} | {preview}...")

    lines.append("")
    lines.append("=== HIT@3 ON LABELED SET ===")
    results = {m: hit_at_k(retriever, m) for m in rc.MODES}
    for i, (q, _) in enumerate(LABELED_SET):
        marks = "  ".join(f"{m.split()[0]}={'Y' if results[m][1][i] else 'n'}" for m in rc.MODES)
        lines.append(f"  {i + 1:>2}. {marks} | {q}")
    for m in rc.MODES:
        lines.append(f"  hit@3 {m:<11}: {results[m][0]:.2f}")
    return "\n".join(lines)


def load_retriever_for_scripts() -> rc.Retriever:
    model = rc.load_embedder()
    saved = rc.load_saved_index()
    chunks, emb = saved if saved is not None else rc.build_and_save_index(model)
    return rc.Retriever(chunks, emb, model)


if __name__ == "__main__":
    print(run_report(load_retriever_for_scripts()))
    print("\nCopy everything above and paste it back into the chat to fill in the README report.")
