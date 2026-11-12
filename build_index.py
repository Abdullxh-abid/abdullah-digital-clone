"""
build_index.py: chunk the documents in data/, embed every chunk ONCE with
all-MiniLM-L6-v2, and save:
    chunks.json     (chunk text + source file + chunk ID + data fingerprint)
    embeddings.npy  (one unit-length 384-d vector per chunk)

Usage:  python build_index.py

Optional: app.py builds these automatically on first launch if they are missing
or out of date, so running this locally is not required.
"""

import rag_core as rc


def main():
    chunks = rc.build_chunks()
    print(f"Loaded {len({c['source'] for c in chunks})} documents -> {len(chunks)} chunks")
    for c in chunks:
        print(f"  {c['chunk_id']:<32} {c['source']:<30} {c['n_words']:>4} words")
    print("Loading embedding model (downloads ~90 MB the first time)...")
    model = rc.load_embedder()
    _, emb = rc.build_and_save_index(model)
    print(f"Embeddings shape: {emb.shape}")


if __name__ == "__main__":
    main()
