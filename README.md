# Digital Clone of Abdullah Bin Abid (Hybrid RAG)

> **This is an AI clone, not the real person.**

A Streamlit chatbot that answers questions as me, **Abdullah Bin Abid (7th semester CS student at UMT)**, using only documents I wrote about myself.

It retrieves relevant passages using **BM25, dense embeddings, or a hybrid of both**, and an LLM generates first-person responses based exclusively on those retrieved passages.

**Live App:** : https://abdullah-digital-clone-cnehxif87nzba9clqw2fnu.streamlit.app

---

## Assignment Information

- **Assignment:** Assignment 2 (CLO-2)
- **Project:** Digital Clone of Abdullah Bin Abid
- **Framework:** Streamlit
- **Deployment Platform:** Streamlit Community Cloud
- **LLM API:** Groq Free API
- **Retrieval Approach:** Hybrid Retrieval-Augmented Generation (RAG)

### Platform Changes from the Original Brief

The following platform changes were made:

- **Deployment:** Streamlit Community Cloud instead of Hugging Face Spaces.
- **LLM API:** Groq Free API instead of the Hugging Face Inference API.

All other UI and RAG requirements remain unchanged.

---

## How It Works

The system follows a Retrieval-Augmented Generation (RAG) pipeline consisting of chunking, retrieval, routing, and response generation.

### 1. Chunking

- Uses paragraph-aware packing to divide documents into manageable chunks.
- Target chunk size: approximately **230 words**.
- Maximum chunk size: **300 words**.
- Overlap between chunks: approximately **40 words**.
- Each chunk retains its `chunk_id` (e.g., `faq-01`) and source file.
- The knowledge base contains **8 files divided into 18 chunks**.

### 2. BM25 Retrieval (Keyword-Based Search)

Uses `rank_bm25.BM25Okapi` for keyword-based retrieval.

Implementation:

- Converts text to lowercase.
- Removes stopwords from tokens.
- Uses BM25 to rank documents based on keyword relevance.
- Returns the top-k matching chunks along with their BM25 scores.

BM25 is particularly useful for exact keywords and rare technical terms.

### 3. Dense Retrieval (Semantic Search)

Uses the `all-MiniLM-L6-v2` sentence-transformer model.

Implementation:

- Generates embeddings for all document chunks.
- Unit-normalises the embeddings.
- Calculates cosine similarity using NumPy.
- Does not require a vector database.

Embeddings are precomputed and stored in:

- `embeddings.npy`
- `chunks.json`

The index can be generated using `build_index.py` or automatically during the first application launch.

Dense retrieval is useful when the question uses different words but has the same underlying meaning as the information in the knowledge base.

### 4. Hybrid Retrieval (Reciprocal Rank Fusion)

The hybrid retrieval method combines the complete BM25 and dense rankings using **Reciprocal Rank Fusion (RRF)**.

The scoring formula is:

\[
RRF(d) = \frac{1}{60 + rank_{BM25}(d)}
+ \frac{1}{60 + rank_{Dense}(d)}
\]

Where:

- `rank_BM25` = Rank of the chunk in BM25 results.
- `rank_Dense` = Rank of the chunk in dense retrieval results.
- `60` = RRF constant.

**Important implementation detail:**

Chunks with a zero BM25 score receive no BM25 contribution.

RRF combines the rankings rather than directly comparing raw scores, which is useful because BM25 scores and cosine similarities operate on completely different scales.

### 5. Query Routing

A small rule-based router handles certain questions before they reach the LLM.

It identifies:

- Private-data requests.
- Questions such as "Are you real?"

These requests receive fixed, predefined safe replies instead of being passed to the LLM.

### 6. Response Generation

The application uses Groq to generate responses.

**Primary model:**
- `openai/gpt-oss-20b`

**Fallback model:**
- `gpt-oss-120b`

The LLM uses the appendix persona prompt and generates first-person answers based only on retrieved passages.

**API key security:**

- The Groq API key is stored in `st.secrets`.
- The key is never hardcoded into the application.

**Fallback mechanism:**

If the Groq API fails, the application displays a templated answer generated from the top retrieved chunk.

---

## User Interface

The Streamlit application includes the following features:

- **Question Box:** Allows users to ask questions about Abdullah.
- **Answer Area:** Displays the generated response.
- **Retrieval Mode Selector:** Allows users to choose between:
  - BM25 Only
  - Dense Only
  - Hybrid
- **Retrieved Chunks Table:** Displays:
  - Source file
  - Chunk ID
  - Retrieval score
  - BM25 rank
  - Dense rank
- **Mode Comparison:** Allows comparison of retrieval results across different modes.
- **AI Clone Banner:** Clearly communicates that the chatbot is an AI clone and not the real person.

---

## Alternative Fusion Method (Not Used)

Another approach considered was **Weighted Min-Max Fusion**.

### How It Works

First, the scores from both retrieval methods are normalised to the range `[0, 1]`.

The normalisation formula is:

\[
s_{norm} = \frac{s - s_{min}}{s_{max} - s_{min}}
\]

The final combined score is:

\[
Score = \alpha \times Dense + (1-\alpha) \times BM25
\]

Where:

- `Dense` = Normalised dense retrieval score.
- `BM25` = Normalised BM25 score.
- `α` = Weight assigned to dense retrieval.

### Why I Did Not Choose It

Although weighted min-max fusion uses score magnitudes, it has some disadvantages:

- A very confident BM25 match can dominate the combined score.
- It requires tuning the alpha parameter.
- It is sensitive to outliers.
- One extremely large BM25 score can squash the other normalised scores.

### Why I Chose RRF Instead

I chose Reciprocal Rank Fusion because:

- It combines rankings rather than raw score magnitudes.
- It requires no alpha tuning.
- It avoids directly comparing BM25 scores with cosine similarities.
- It provides a straightforward way to combine keyword-based and semantic retrieval.

---

## Knowledge Base

The knowledge base consists of **8 Markdown files containing approximately 3,200 words**, all written in my first-person voice.

These documents provide the information used by the chatbot to answer questions about me.

### Knowledge Base Files

| File | Purpose |
|---|---|
| `bio.md` | Personal biography and factual information |
| `cv.md` | CV and factual information |
| `personality_and_opinions.md` | Personality, opinions, and preferences |
| `how_i_work_and_advice.md` | Working style, study habits, and advice |
| `plant_disease_writeup.md` | Long-form technical write-up (~880 words) |
| `project_3d_food_ar.md` | Project information |
| `project_game_hub.md` | Project information |
| `faq.md` | 30 first-person question-and-answer pairs |

### Why These Documents Were Selected

I selected these documents because they cover the information classmates would realistically ask me about, including:

- Academic studies.
- University projects.
- WPBrigade internship.
- Hobbies.
- Personal opinions.
- Personality.
- Study habits and working style.

The FAQ document also captures my natural speaking style, including:

- Casual language.
- Common idioms.
- Occasional emojis.

### Supporting Both Retrieval Methods

The knowledge base was deliberately written to support both keyword-based and semantic retrieval.

**For BM25:**

Rare keywords such as:

- WPBrigade
- PlantVillage
- MobileNetV2

give BM25 specific terms to match.

**For Dense Retrieval:**

Six of the eight files end with an **"In plain words"** paragraph.

These paragraphs explain the same information without repeating the exact technical keywords, allowing dense retrieval to find relevant information through semantic similarity.

### Privacy Considerations

The knowledge base does not include:

- Phone numbers.
- Addresses.
- Personal IDs.
- Details about other people.

---

## Retrieval Evaluation

I evaluated the three retrieval modes using **five test questions**, retrieving the top 3 chunks for each mode.

The results were generated using `eval_report.py`.

### Evaluation Results

The following table shows the top retrieval result for each mode and the best-performing mode for each question.

| Question | BM25 Result | Dense Result | Hybrid Result | Best Mode |
|---|---|---|---|---|
| Q1: What did you do at WPBrigade? | `faq-00` (2.57), `cv-00` (2.32), `bio-00` (0.00) | `cv-00` (0.33), `how_i_work_and_advice-00` (0.19), `bio-00` (0.18) | `cv-00` (0.0325, b2/d1), `faq-00` (0.0311, b1/d8), `how_i_work_and_advice-00` (0.0161, -/d2) | Hybrid |
| Q2: When are you most productive? | `bio-00`, `bio-01`, `cv-00` (all 0.00) | `how_i_work_and_advice-00` (0.33), `personality_and_opinions-00` (0.28), `faq-02` (0.26) | Same three chunks as dense, in the same order | Dense (Hybrid ties) |
| Q3: Which football club do you cheer for? | `personality_and_opinions-00` (7.55), `faq-02` (1.83), `faq-01` (1.44) | `faq-02` (0.13), `bio-01` (0.08), `personality_and_opinions-00` (0.06) | `faq-02` (0.0325, b2/d1), `personality_and_opinions-00` (0.0323, b1/d3), `faq-01` (0.0294, b3/d14) | BM25 |
| Q4: How accurate was your MobileNetV2 model on PlantVillage? | `faq-01` (4.10), `plant_disease_writeup-01` (1.91), `plant_disease_writeup-00` (1.72) | `plant_disease_writeup-02` (0.54), `plant_disease_writeup-03` (0.52), `cv-02` (0.51) | `faq-01` (0.0318, b1/d5), `plant_disease_writeup-02` (0.0313, b7/d1), `plant_disease_writeup-01` (0.0313, b2/d6) | Hybrid |
| Q5: What is your view on AI being used for creative work? | `personality_and_opinions-00` (3.24), `cv-00` (2.45), `how_i_work_and_advice-00` (2.33) | `personality_and_opinions-00` (0.49), `cv-00` (0.37), `faq-02` (0.33) | `personality_and_opinions-00` (0.0328, b1/d1), `cv-00` (0.0323, b2/d2), `faq-02` (0.0315, b4/d3) | Tie |

**Notation:**

- BM25 scores are BM25Okapi scores.
- Dense scores are cosine similarities.
- Hybrid scores are RRF scores.
- `b` represents BM25 rank.
- `d` represents dense rank.
- `-` means the chunk received a zero BM25 score.
- Chunks are approximately 230 words, so a single chunk can contain several FAQ answers.

### Q1: What did you do at WPBrigade?

**Best mode: Hybrid**

This question contains an exact, rare keyword.

- BM25 found both chunks containing the keyword.
- Dense retrieval missed `faq-00`, placing it at dense rank 8.
- Hybrid rescued `faq-00` and ranked it second while keeping the CV chunk first.

The third result is dense retrieval noise.

Hybrid performed best because it combined the strong keyword match from BM25 with the relevant results from dense retrieval.

### Q2: When are you most productive?

**Best mode: Dense (Hybrid ties)**

This question is a paraphrase and does not share exact words with the knowledge base.

- The word "productive" does not appear in my notes.
- BM25 returned zero scores for every chunk.
- Its top three results were simply the first chunks in file order.
- Dense retrieval correctly understood the meaning of the question and retrieved information about my working habits, including being a "night owl".

Hybrid produced the same results as dense retrieval because RRF ignores zero-score BM25 chunks.

### Q3: Which football club do you cheer for?

**Best mode: BM25**

The word "football" appears in both the query and the relevant personality document.

- BM25 ranked `personality_and_opinions-00` first with a score of 7.55.
- The next result had a score of 1.83.
- Dense retrieval ranked the relevant personality chunk third.
- Hybrid included the relevant chunk in its top three but ranked it second.

BM25 performed best because the exact keyword match produced a strong retrieval score.

### Q4: How accurate was your MobileNetV2 model on PlantVillage?

**Best mode: Hybrid**

This question combines technical keywords with semantic meaning.

- BM25 strongly matched the rare keywords.
- However, it ranked the transfer-learning chunk (`plant_disease_writeup-02`) seventh.
- Dense retrieval ranked that chunk first but missed the FAQ chunk.
- Hybrid retrieved both the FAQ chunk and the transfer-learning chunk in its top three, along with the dataset chunk.

Hybrid was the only mode that successfully combined these relevant chunks in its top three results.

### Q5: What is your view on AI being used for creative work?

**Best mode: Tie**

All three retrieval methods ranked `personality_and_opinions-00` first.

Hybrid was particularly trustworthy because both retrieval methods independently agreed on the same top result:

- BM25 rank: 1
- Dense rank: 1

Generic words such as "work" and "used" also retrieved `cv-00`, but the relevant opinions chunk remained at the top.

---

## Overall Retrieval Performance

I also evaluated all three retrieval methods using a **10-question hand-labelled test set**.

A hit means that at least one of the top three retrieved chunks contains the answer text.

| Retrieval Method | Hit@3 |
|---|---:|
| BM25 | 0.90 |
| Dense | 1.00 |
| Hybrid | 1.00 |

### Observations

- BM25 achieved a hit@3 score of 0.90.
- Dense retrieval achieved a perfect hit@3 score of 1.00.
- Hybrid retrieval also achieved a perfect hit@3 score of 1.00.
- BM25's only miss was the paraphrased question: "When are you most productive?"
- Hybrid performed at least as well as the best individual retrieval method.

These results demonstrate the advantage of combining keyword-based and semantic retrieval.

---

## End-to-End Testing

The application was tested using `test_clone.py`.

<pre>
15 PASS
0 FAIL
</pre>

### Test Coverage

**1. Questions Answered Correctly**

Five questions whose answers exist in the knowledge base were tested.

These included:

- UMT.
- Barcelona.
- 94.45% validation accuracy.
- Claude and Cursor.
- Master's in Canada.

All five were answered correctly in the first person.

**2. Questions Not Covered by the Knowledge Base**

Three questions were tested where the answers were not present in the documents:

- CGPA.
- Favourite colour.
- Japan.

The chatbot correctly responded:

> I don't have that in my notes.

**3. Privacy Protection**

Two private-data requests were tested.

Both were refused through the rule-based safety router.

**4. Identity Protection**

The question:

> Are you the real Abdullah?

was tested.

The chatbot correctly identified itself as an AI clone rather than the real person.

---

## Failure Cases and Limitations

Although the hybrid approach performed well, some limitations were identified during evaluation.

### 1. BM25 Failure: Paraphrased Questions (Q2)

**Question:** When are you most productive?

The answer ("I'm a night owl") exists in `how_i_work_and_advice.md`.

However, after stopword removal, the only remaining query token is `productive`, which does not appear anywhere in the knowledge base.

As a result:

- Every chunk receives a BM25 score of 0.
- BM25's top three results are simply the first chunks in file order:
  - `bio-00`
  - `bio-01`
  - `cv-00`
- None of these chunks mention my working hours.
- The LLM receives irrelevant context and cannot answer correctly, even though the answer exists in my notes.

**Why does this happen?**

This demonstrates the main weakness of lexical search.

BM25 cannot naturally connect semantically related expressions such as:

- "productive"
- "night owl"
- "late-night person"

Dense retrieval understands the semantic relationship and successfully retrieves the relevant information.

Hybrid retrieval also inherits this advantage because RRF ignores zero-score BM25 chunks.

### 2. Hybrid Retrieval Limitation: Football Question (Q3)

**Question:** Which football club do you cheer for?

Hybrid retrieval ranked `faq-02` above `personality_and_opinions-00`, even though BM25 gave the personality chunk a much stronger score.

The results were:

| Chunk | BM25 Rank | Dense Rank | RRF Score |
|---|---:|---:|---:|
| `faq-02` | 2 | 1 | 0.0325 |
| `personality_and_opinions-00` | 1 | 3 | 0.0323 |

The difference is small.

RRF only considers rankings, not the magnitude of the original scores.

Therefore:

- The dense rank of 1 and BM25 rank of 2 slightly outweighed the BM25 rank of 1 and dense rank of 3.
- The strong BM25 score difference (7.55 versus 1.83) was discarded.

The answer was still correct because the relevant chunk appeared in the top three.

However, this demonstrates a limitation of rank-only fusion: it cannot distinguish between a very confident match and a marginal match.

---

## Future Improvements

One improvement planned for the next version is adding a **cross-encoder reranker**.

### 1. Cross-Encoder Reranking

A possible model is:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The idea is to:

1. Retrieve the top 10 chunks using hybrid retrieval.
2. Pass these chunks through the cross-encoder.
3. Evaluate each question and passage together.
4. Re-rank the retrieved chunks based on relevance.

Unlike the current independent embedding approach, a cross-encoder evaluates the question and passage together.

### 2. Smaller Chunks

Another improvement is reducing the chunk size from approximately 230 words to approximately 150 words.

The reason is that `all-MiniLM-L6-v2` only reads 256 word-pieces (approximately 190 words).

As a result, the tail of longer chunks can be ignored by the dense retriever.

Smaller chunks would:

- Reduce truncation.
- Improve retrieval precision.
- Keep relevant information within the embedding model's input limit.

### Expected Improvement

Combining smaller chunks with a cross-encoder reranker should help address:

- Dense retrieval truncation.
- Near-tie rankings such as Q3.
- Retrieval of less relevant passages.

---

## Project Structure

The repository is organised into the following files and directories:

```text
Digital-Clone/
│
├── app.py
│   └── Streamlit user interface
│
├── rag_core.py
│   └── Chunking, BM25, dense retrieval,
│       RRF, routing, Groq API call, and fallback
│
├── build_index.py
│   └── Builds chunks.json and embeddings.npy
│
├── eval_report.py
│   └── Prints top-3 results for each retrieval mode
│       and calculates hit@3
│
├── test_clone.py
│   └── PASS/FAIL checks for facts, unknowns,
│       privacy, identity, and hit@3 requirements
│
├── requirements.txt
│   └── Pinned dependencies for Python 3.11
│
├── secrets.toml.example
│   └── Template for the GROQ_API_KEY secret
│
├── data/
│   └── Eight Markdown knowledge-base documents
│
├── chunks.json
│   └── Generated document chunks
│
├── embeddings.npy
│   └── Precomputed dense embeddings
│
└── DEPLOYMENT.md
    └── Setup, deployment, manual testing plan,
        and deployment checklist
```

**Additional notes:**

- `requirements.txt` is pinned for Python 3.11.
- It includes Linux CPU PyTorch dependencies for deployment.
- It also includes Intel Mac PyTorch 2.2.2.
- The real API secrets are Git-ignored.
- `secrets.toml.example` provides a template for configuring the Groq API key.

---

## Running Locally

### 1. Install Dependencies

Make sure Python 3.11 is installed.

Run:

```bash
pip install -r requirements.txt
```

### 2. Configure Groq API Key

Create your Streamlit secrets configuration and add your Groq API key.

The application reads the key through:

```python
st.secrets
```

Use `secrets.toml.example` as the configuration template.

Never commit your real API key to GitHub.

### 3. Build the Retrieval Index (Optional)

Run:

```bash
python build_index.py
```

This generates:

- `chunks.json`
- `embeddings.npy`

The application can also generate the index automatically during its first launch.

### 4. Run the Application

Execute:

```bash
streamlit run app.py
```

The application will start locally and can be accessed through the Streamlit URL displayed in the terminal.

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

For complete deployment instructions, configuration details, manual testing steps, and the deployment checklist, refer to:

`DEPLOYMENT.md`

---

## Final Notes

This project demonstrates the implementation of a personal AI digital clone using Retrieval-Augmented Generation.

It combines:

- Keyword-based retrieval using BM25.
- Semantic retrieval using dense embeddings.
- Hybrid retrieval using Reciprocal Rank Fusion.
- Rule-based safety routing.
- LLM-based first-person response generation.
- Groq API integration.
- Streamlit deployment.
- Retrieval evaluation and end-to-end testing.

The main objective is to ensure that the chatbot responds naturally as Abdullah while remaining grounded in the information contained in the provided personal knowledge base.

**This is an AI clone, not the real Abdullah Bin Abid.**
