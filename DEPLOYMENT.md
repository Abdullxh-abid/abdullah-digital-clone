# Deployment Guide (beginner friendly)

Follow the parts in order. Total time: about 30 to 45 minutes, most of it waiting for installs.

---

## Part A. Get a free Groq API key (2 minutes)

1. Go to **https://console.groq.com** and sign up with Google or email. No credit card is needed for the free plan.
2. In the left menu, open **API Keys** and click **Create API Key**. Name it `abdullah-clone`.
3. Copy the key (it starts with `gsk_`). **You only see it once**, so paste it into a private note for now.
4. Never paste the key into any `.py` file, the README or GitHub. It only goes into Streamlit **Secrets** (Part D) or your local `.streamlit/secrets.toml` (Part B), which is git-ignored.

> Free plan limits for `openai/gpt-oss-20b`: about 30 requests per minute and 1,000 requests per day, which is plenty for a class demo. Groq's current list of free models is at console.groq.com/docs/models.

---

## Part B. Run it locally first (Intel MacBook)

1. **Install Python 3.11** from https://www.python.org/downloads/macos/ (pick the latest 3.11.x "macOS 64-bit universal2 installer"). After installing, double-click **Install Certificates.command** in `Applications/Python 3.11`. This prevents SSL errors when the embedding model downloads.
2. Unzip `abdullah-clone.zip`, open **Terminal**, and go into the folder:
   ```bash
   cd ~/Downloads/abdullah-clone
   ```
3. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
4. Install the pinned packages (takes 3 to 8 minutes):
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Add your key for local runs:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   open -e .streamlit/secrets.toml
   ```
   Replace `gsk_paste_your_key_here` with your real key, then save.
6. (Optional) Build the index: `python build_index.py`. It downloads the ~90 MB model once, then prints 18 chunks and `Embeddings shape: (18, 384)`. You can skip this step, because the app builds the index on first launch.
7. Start the app:
   ```bash
   streamlit run app.py
   ```
   Your browser opens at http://localhost:8501. Ask: *What did you do at WPBrigade?*
8. Run the automated checks and the report script:
   ```bash
   python test_clone.py
   python eval_report.py
   ```
   **Copy the full output of `eval_report.py` (and of `test_clone.py`) and paste it back to Claude** to fill in the README report table, the hit@3 line and the second failure case.

### If installation fails on the Intel Mac (fallbacks, try in order)

| Problem | Fix |
|---|---|
| `No matching distribution found for torch==2.2.2` | You're on the wrong Python. Run `python --version` inside the venv: it must be **3.11** (3.8 to 3.12 also work). Delete `.venv` and redo step 3 with `python3.11`. |
| torch still won't install | `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`, then run `pip install -r requirements.txt` again. |
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2` | `pip install "numpy==1.26.4" --force-reinstall` |
| `SSL: CERTIFICATE_VERIFY_FAILED` when downloading the model | Run `/Applications/Python\ 3.11/Install\ Certificates.command`, then retry. |
| `clang` / "building wheel" errors | You got a source build. Run `pip install --only-binary=:all: -r requirements.txt` to force prebuilt wheels. |
| Nothing works locally | **Skip local entirely.** Deploy (Parts C and D); the app builds embeddings itself. Get the report from the app sidebar via **Run evaluation report**. To run `test_clone.py`, use Google Colab: upload the zip, then run `!unzip abdullah-clone.zip -d . && cd abdullah-clone && pip install rank-bm25 sentence-transformers streamlit` and `!cd abdullah-clone && GROQ_API_KEY=gsk_... python test_clone.py` |

---

## Part C. Put the code on GitHub

1. Create a free account at https://github.com if you don't have one.
2. Click **+** (top right) > **New repository**. Name: `abdullah-digital-clone`. Choose **Public**. Don't add a README (we have one). Click **Create repository**.
3. On the new repo page click **uploading an existing file**.
4. In Finder, open the unzipped `abdullah-clone` folder and press **Cmd + Shift + .** to show hidden files (`.gitignore`, `.streamlit`).
5. Drag **everything inside the folder** into the GitHub upload box: `app.py`, `rag_core.py`, `build_index.py`, `eval_report.py`, `test_clone.py`, `requirements.txt`, `README.md`, `DEPLOYMENT.md`, `secrets.toml.example`, `.gitignore`, the `data` folder and the `.streamlit` folder.
6. ⚠️ **Do NOT upload** `.streamlit/secrets.toml` (your real key) or the `.venv` folder. If you ran the app locally you may also upload `chunks.json` and `embeddings.npy` (optional; this skips the first-launch build).
7. Click **Commit changes**. Check that `data/` shows 8 `.md` files and that there is **no** `secrets.toml` (only `secrets.toml.example`).

---

## Part D. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and click **Continue with GitHub**. Authorize Streamlit.
2. Click **Create app** > **Deploy a public app from GitHub** (or "Yup, I have an app").
3. Fill in:
   - **Repository:** `your-username/abdullah-digital-clone`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** something like `abdullah-clone` (gives `abdullah-clone.streamlit.app`)
4. Click **Advanced settings**:
   - **Python version: 3.11** ← required (the Linux torch wheel in `requirements.txt` is built for 3.11).
   - **Secrets:** paste exactly:
     ```toml
     GROQ_API_KEY = "gsk_your_real_key"
     ```
   - Click **Save**.
5. Click **Deploy**. The first build takes about 3 to 6 minutes. The first page load then takes about 1 minute more while it downloads the model and builds the embeddings (the spinner says so). Later visits are fast because of `st.cache_resource`.
6. Forgot the secret, or need to change it? Open the app > **⋮ menu (top right) > Settings > Secrets**, edit, **Save**. The app restarts by itself.

## Part E. Make sure the app is public

1. Apps deployed from a public repo are public by default. To confirm, open your app > **⋮ > Settings > Sharing** and check that **Who can view this app** is **"This app is public and searchable"** (or "Anyone with the link").
2. Test in a **private / incognito browser window** while logged out. The app must open without any login.
3. Paste the URL into `README.md` (line "Live app") on GitHub (pencil icon > edit > Commit).
4. Note: free apps go to sleep after a few days without visits. Open the link before your demo and click **"Yes, get this app back up!"** if you see it.

## Part F. Common build errors on Streamlit Cloud

| Error in the build log | Fix |
|---|---|
| `torch-2.2.2+cpu-cp311... is not a supported wheel on this platform` | Python version isn't 3.11. Delete the app and redeploy with **Python 3.11** in Advanced settings (the version can't be changed after deploying). |
| Build fails while downloading the torch URL | Replace the first `torch @ https://...` line in `requirements.txt` with `torch==2.2.2 ; sys_platform == "linux"` (bigger download, but works). Commit; the app rebuilds. |
| `ModuleNotFoundError: No module named 'rag_core'` | `rag_core.py` wasn't uploaded. Upload it next to `app.py`. |
| `FileNotFoundError: No .md or .txt files found in data` | The `data` folder is missing or misnamed (must be lowercase `data`). |
| App shows "No GROQ_API_KEY in secrets" | Add the secret (Part D step 6). The name must be exactly `GROQ_API_KEY`. |
| "The API key was rejected" | Key copied wrong or deleted. Create a new one in Groq and update Secrets. |
| "rate limit reached" | Free-tier limit. Wait about a minute. The app still shows retrieved chunks and a templated answer. |
| "Model ... is not available" | Groq renamed models. Add `GROQ_MODEL = "openai/gpt-oss-120b"` (or a current free model from console.groq.com/docs/models) to Secrets. |
| App crashes with "resource limits" | Reboot from **⋮ > Reboot app**. The app uses roughly 1 GB of RAM, which normally fits. |
| Red "Dense model could not load" in the sidebar | Temporary Hugging Face download problem. **⋮ > Reboot app**. BM25 keeps working meanwhile. |

---

## Part G. How to test manually (in the live app)

Type each question, try **each mode**, and compare with "correct looks like". In every mode the sources table must show **source file, chunk ID and score**, and the red banner must be visible. Chunk IDs marked *(verified)* came from a real BM25 run. The others are what you should expect; confirm them yourself.

| # | Question | What a correct result looks like |
|---|---|---|
| 1 | What did you do at WPBrigade? | **BM25:** `faq-00` and `cv-00` at the top *(verified)*. **Dense/Hybrid:** an FAQ or CV chunk about the internship in the top 3. **Answer:** an AI research internship working on AI models for foreign companies. |
| 2 | When are you most productive? | **BM25:** all scores 0.00, irrelevant chunks *(verified)*, so a weak or "not in my notes" answer. **Dense/Hybrid:** `how_i_work_and_advice-00` or the FAQ night-owl chunk near the top. **Answer:** "I'm a night owl." This is the dense-wins example. |
| 3 | Which football club do you cheer for? | `personality_and_opinions-00` top in BM25 *(verified)*. **Answer:** Barcelona ⚽. |
| 4 | How accurate was your MobileNetV2 model on PlantVillage? | BM25 top `faq-01`, then plant write-up chunks *(verified)*. **Answer:** 94.45% validation accuracy, 91% after epoch 1. |
| 5 | Can diners see their meal in 3D before ordering? | Dense/Hybrid should surface `project_3d_food_ar-00`. **Answer:** yes, via a QR code and life-size AR dishes; he was team lead. |
| 6 | What do you think about AI in art? | `personality_and_opinions-00` near the top. **Answer:** AI should be banned in self-expressive art and only used for repetitive work. |
| 7 | What is your CGPA? | Any chunks. **Answer must be exactly:** "I don't have that in my notes." |
| 8 | What is your phone number? | **Answer:** a polite refusal ("Sorry, I can't share that..."), with no digits. The caption says *Router: private*. |
| 9 | Are you the real Abdullah? | **Answer:** "No, I'm not the real person. I'm an AI clone of Abdullah Bin Abid..." The caption says *Router: identity*. |
| 10 | What advice would you give a junior? | `how_i_work_and_advice-00` or the FAQ. **Answer:** do 2 to 3 projects that solve significant modern problems. |

Also check: with **"Also show the top-3 for every mode"** ticked, Hybrid's list should differ from at least one of the other two on questions 1, 2 and 5. The sidebar **Run evaluation report** button prints the full report.

---

## Part H. Pre-submission checklist

**Rubric**
- [ ] **1. Knowledge base (0.5):** 8 files in `data/` with bio/CV, personality, long-form write-up and a 30-question FAQ. The sidebar shows 18 chunks; chunk IDs and sources are visible.
- [ ] **2. BM25 + dense (1.0):** both modes return top-k with scores; `embeddings.npy` + `chunks.json` are saved (built by `build_index.py` or on first launch).
- [ ] **3. Hybrid (1.0):** RRF formula `1/(60+rank)`; the comparison view shows that the Hybrid ranking differs.
- [ ] **4. Persona + grounding (0.75):** first person; "I don't have that in my notes." for unknowns; the key is only in Streamlit Secrets; `test_clone.py` passes.
- [ ] **5. App + deployment (0.75):** the public URL opens in incognito, the mode selector works, the sources panel is visible, the banner is visible.
- [ ] **6. Report (1.0):** README table filled with the **real** `eval_report.py` output for all 5 questions, a best mode for each, a failure case, an improvement, and a hit@3 line.
- [ ] The live URL is pasted into the README and submitted.

**"Do not include" list (minus 1 mark if broken)**
- [ ] No phone numbers, home address, CNIC, passport or ID numbers anywhere (`data/`, README, code).
- [ ] No passwords, bank or account details. **No API key in any file on GitHub** (search the repo for `gsk_`: you should only find the placeholder in `secrets.toml.example`).
- [ ] No private details about family, friends or classmates.
- [ ] No WhatsApp or chat logs, and no copyrighted text.
- [ ] Abdullah has read every file in `data/` and agrees it can be public.
