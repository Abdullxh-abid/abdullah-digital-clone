Digital Clone of Abdullah Bin Abid (Hybrid RAG)
This is an AI clone, not the real person.
Live app: https://abdullah-digital-clone-cnehxif87nzba9clqw2fnu.streamlit.app

A Streamlit chatbot that answers questions as me, Abdullah Bin Abid (7th semester CS student at UMT), using only documents I
wrote about myself. It retrieves passages with BM25, dense embeddings or a hybrid of both, and an LLM replies in first person
from those passages only.
Assignment 2 (CLO-2). Platform changes from the brief: Streamlit Community Cloud instead of Hugging Face Spaces, and the
Groq free API instead of the HF Inference API. All UI and RAG requirements are unchanged.
How it works
Step Implementation
Chunking Paragraph-aware packing to ~230 words (max 300) with ~40-word overlap; each chunk keeps chunk_id (e.g. faq-01 ). 8 files → 18 chunks.
source file and
BM25 rank_bm25.BM25Okapi on lowercased, stopword-filtered tokens. Returns top-k with scores.
Dense
all-MiniLM-L6-v2 embeddings, unit-normalised, cosine similarity with numpy (no vector DB). Precomputed once into
embeddings.npy + chunks.json ( build_index.py , or automatically on first app launch).
Hybrid Reciprocal Rank Fusion over the full BM25 and dense rankings: score = 1/(60 + rank_bm25) + 1/(60 +
rank_dense) . Chunks with zero BM25 score get no BM25 term.
Routing A small rule-based router sends private-data requests and "are you real?" questions to fixed safe replies before the
LLM.
Generation Groq openai/gpt-oss-20b (fallback gpt-oss-120b ) with the appendix persona prompt. Key read from st.secrets ,
never in code. If the API fails, the app shows a templated answer from the top chunk.
UI Question box, answer area, mode selector (BM25 only / Dense only / Hybrid), retrieved-chunk table (source, chunk ID,
score, BM25 rank, dense rank), mode comparison, and the AI-clone banner.
Alternative fusion (not used): weighted min-max. Min-max normalise each retriever's scores to [0, 1] ( (s - min) / (max -
min) ), then compute alpha * dense + (1 - alpha) * bm25 . It uses the score magnitudes, so a very confident BM25 match can
dominate. However it needs a tuned alpha and is sensitive to outliers, because one huge BM25 score squashes the others. I
chose RRF because it only uses ranks, needs no tuning, and BM25 scores and cosine similarities live on completely different
scales.
Report
Knowledge base and why
8 Markdown files (~3,200 words) written in my first-person voice: bio.md and cv.md (factual), personality_and_opinions.md
and how_i_work_and_advice.md (personality, opinions, study habits, advice), plant_disease_writeup.md (long-form technical
write-up, ~880 words), project_3d_food_ar.md and project_game_hub.md (project pages), and faq.md (30 first-person Q&As). I
chose these because they cover what classmates would actually ask me: studies, projects, the WPBrigade internship, hobbies and
opinions. The FAQ carries my speaking style (casual, idioms, occasional emoji). Rare keywords such as WPBrigade, PlantVillage
and MobileNetV2 give BM25 something to win on. Six of the eight files end with an "In plain words" paragraph that avoids those
keywords, so dense search can win on paraphrased questions. No phone numbers, addresses, IDs or details about other people
are included.
Five test questions (top-3 per mode, from eval_report.py )
Scores: BM25 = BM25Okapi score, Dense = cosine similarity, Hybrid = RRF score ( bm25_rank, dense_rank "-" means BM25 scored the chunk 0. Chunks are ~230 words, so one chunk can hold several FAQ answers.
in brackets).

# Question BM25 result Dense result Hybrid result
Best mode
and why
Q1
What did
you do at
WPBrigade?
(exact rare
keyword)
faq-00 (2.57), cv-00
(2.32), then bio-00 (0.00,
filler). Only 2 chunks contain
the word.
cv-00 (0.33),
how_i_work_and_advice-00
(0.19), bio-00 (0.18). faq-
00 is missing from the top 3
(dense rank 8).
cv-00 (0.0325, b2/d1), faq-
00 (0.0311, b1/d8),
how_i_work_and_advice-00
(0.0161, -/d2)
Hybrid
(BM25 is
nearly as
good). BM25
found both
chunks that
contain the
keyword, and
dense
dropped
faq-00.
Hybrid
rescued faq-
00 to rank 2
while keeping
the CV chunk
on top. Its
third slot is
dense noise.
Q2
When are
you most
productive?
(paraphrase,
no shared
words)
bio-00 , bio-01 , cv-00 , all
scores 0.00. "productive"
appears nowhere in my notes,
so this is just file order.
how_i_work_and_advice-00
(0.33),
personality_and_opinions-
00 (0.28), faq-02 (0.26)
Same three chunks as dense,
in the same order (bm25_
rank
"-" for all)
Dense
(hybrid ties).
Dense
matched the
meaning
(when I work,
"night owl")
with zero
shared words.
BM25 had
nothing to
rank. Hybrid
inherits the
dense list
because RRF
ignores zero-
score BM25
chunks.
Q3
Which
football club
do you
cheer for?
personality_and_opinions-
00 (7.55), faq-02 (1.83),
faq-01 (1.44)
faq-02 (0.13), bio-01
(0.08),
personality_and_opinions-
00 (0.06)
faq-02 (0.0325, b2/d1),
personality_and_opinions-
00 (0.0323, b1/d3), faq-01
(0.0294, b3/d14)
BM25.
"football" is in
the query and
the
personality
chunk, so it
ranked that
chunk first by
a wide margin
(7.55 vs 1.83).
Dense put it
third, behind
a food chunk.
Hybrid got it
into the top 3
but ranked it
second.
Q4
How
accurate
was your
MobileNetV2
model on
PlantVillage?
(keyword +
meaning)
faq-01 (4.10),
plant_disease_writeup-01
(1.91),
plant_disease_writeup-00
(1.72)
plant_disease_writeup-02
(0.54),
plant_disease_writeup-03
(0.52), cv-02 (0.51). faq-
01 is only dense rank 5.
faq-01 (0.0318, b1/d5),
plant_disease_writeup-02
(0.0313, b7/d1),
plant_disease_writeup-01
(0.0313, b2/d6)
Hybrid. Two
rare keywords
make BM25
strong, but
BM25 ranked
the transfer-
learning
chunk ( -02 )
only 7th, and
dense missed
the FAQ
chunk. Hybrid
is the only list
with the FAQ
chunk and
-02
together, plus
the dataset
chunk -01.
Q5
What is your
view on AI
being used
for creative
work?
(opinion)
personality_and_opinions-
00 (3.24), cv-00 (2.45),
how_i_work_and_advice-00
(2.33)
personality_and_opinions-
00 (0.49), cv-00 (0.37),
faq-02 (0.33)
personality_and_opinions-
00 (0.0328, b1/d1), cv-00
(0.0323, b2/d2), faq-02
(0.0315, b4/d3)
Tie. All three
put the
opinions
chunk first.
Hybrid is the
most
trustworthy,
since both
retrievers
independently
agree (rank 1
and 1).
Generic
words like
"work" and
"used" pull in
cv-00.
hit@3 on a 10-question hand-labeled set (a hit means a top-3 chunk contains the answer text): BM25 0.90 · Dense 1.00 ·
Hybrid 1.00. BM25's only miss was the paraphrased "When are you most productive?" Hybrid was at least as good as the best
single retriever.
End-to-end checks ( test_clone.py ): 15 PASS, 0 FAIL. 5 in-notes questions answered correctly in first person (UMT, Barcelona,
94.45 % validation accuracy, Claude and Cursor, master's in Canada); 3 not-in-notes questions (CGPA, favourite colour, Japan)
answered "I don't have that in my notes."; 2 private-data requests refused; "Are you the real Abdullah?" answered as an AI clone.
Failure case
Q2, BM25 only: "When are you most productive?" The answer ("I'm a night owl") is in how_i_work_and_advice.md , but after
stopword removal the only query token is productive , which never appears in my notes. Every chunk scores 0, so BM25's "top 3"
are just the first chunks in file order ( bio-00 , bio-01 , cv-00 ). None of them mention my working hours, so in BM25 mode the
LLM gets irrelevant context and cannot answer correctly, even though the fact is in my notes. This is the textbook weakness of
lexical search: it can't connect productive with night owl or late-night person. Dense retrieval captures that meaning, and hybrid
inherits it because RRF ignores zero-score BM25 chunks.
A weaker hybrid result (Q3): for "Which football club do you cheer for?", hybrid ranked faq-02 (a food-and-favourites chunk)
above personality_and_opinions-00 , the chunk BM25 scored far higher (7.55 vs 1.83). RRF only uses ranks, so dense rank 1 plus
BM25 rank 2 (0.0325) narrowly beat BM25 rank 1 plus dense rank 3 (0.0323), and the strong BM25 score margin was thrown away.
The answer was still correct because both chunks were in the top 3, but it shows the cost of rank-only fusion: it can't tell a
confident match from a marginal one.
Improvement for next time
Add a cross-encoder reranker (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2 ) over the top-10 hybrid results, and shrink chunks
to ~150 words. all-MiniLM-L6-v2 only reads 256 word-pieces (about 190 words), so the tail of my longer chunks is ignored by
the dense retriever. Smaller chunks plus a reranker that reads the full question and passage together should fix both truncation
and near-tie rankings like Q3.
Repository
app.py Streamlit UI
rag_core.py chunking, BM25, dense, RRF, router, Groq call, fallback
build_index.py builds chunks.json + embeddings.npy (optional; app does it on first launch)
eval_report.py prints top-3 per mode for the report questions + hit@3
test_clone.py PASS/FAIL checks: facts, unknowns, privacy, identity, hit@3
requirements.txt pinned for Python 3.11 (Linux CPU torch / Intel Mac torch 2.2.2)
secrets.toml.example template for the GROQ_API_KEY secret (real secrets are git-ignored)
data/ 8 knowledge-base documents
DEPLOYMENT.md step-by-step setup, deployment, manual test plan and checklist
Run locally: pip install -r requirements.txt then streamlit run app.py . See DEPLOYMENT.md for full steps.
