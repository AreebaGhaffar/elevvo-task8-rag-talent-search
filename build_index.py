"""Build the resume vector index.

Usage:
  python build_index.py [--rebuild] [--faiss]

Default builds the ChromaDB collection at chroma_db/ (220 resumes ->
~600-900 chunks, a few minutes on CPU for the first build).
--faiss additionally builds the FAISS fallback index at outputs/faiss_index/.

The collection is persistent: re-running without --rebuild reuses it.
"""
import argparse
import time

from src import resume_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--faiss", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    docs = resume_store.build_chunks()
    print(f"{len(docs)} chunks from 220 resumes "
          f"({len(set(d['resume_id'] for d in docs))} resumes)")
    collection = resume_store.build_chroma(docs, rebuild=args.rebuild)
    print(f"ChromaDB collection '{collection.name}': "
          f"{collection.count()} chunks in {time.time() - t0:.0f}s")
    if args.faiss:
        resume_store.build_faiss_index(docs)


if __name__ == "__main__":
    main()
