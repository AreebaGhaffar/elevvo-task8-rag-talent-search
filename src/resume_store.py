"""Resume chunking, embedding and vector-store access for Task 8.

Design (CPU-friendly, ~6 GB RAM):
- One document per resume, split into section chunks (WORK EXPERIENCE,
  EDUCATION, SKILLS, ...) with a sliding-window fallback for long sections.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim, 90 MB),
  normalised, cosine similarity.
- Primary store: ChromaDB, persistent collection at chroma_db/
  (gitignored; rebuild with build_index.py).
- Fallback store: FAISS (faiss-cpu) flat inner-product index + a JSON
  sidecar of chunk metadata — build with build_faiss_index().

Chunk metadata carried in the store: resume_id, name, designation,
skills, companies, degree, college, grad_year, years_experience,
location, section. Retrieval aggregates chunk hits per resume by the
best (max) chunk score.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RESUMES_CSV = ROOT / "results" / "resumes.csv"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "resumes"
EMBED_MODEL = "models/sentence-transformers/all-MiniLM-L6-v2"

SECTION_HEADERS = [
    "WORK EXPERIENCE", "EXPERIENCE", "EDUCATION", "SKILLS",
    "ADDITIONAL INFORMATION", "TECHNICAL SKILLS", "PROJECTS",
    "CERTIFICATIONS", "SUMMARY", "OBJECTIVE", "INTERESTS",
]

CHUNK_CHARS = 900   # ~150 words per chunk
CHUNK_OVERLAP = 150


def load_resumes() -> pd.DataFrame:
    df = pd.read_csv(RESUMES_CSV)
    return df.fillna("")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split resume text into (section, text) pairs on header lines."""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current = "GENERAL"
    buf: list[str] = []
    for line in lines:
        stripped = line.strip().upper()
        header = next((h for h in SECTION_HEADERS if stripped == h or stripped.startswith(h + " ")), None)
        if header and buf:
            sections.append((current, "\n".join(buf)))
            current, buf = header, []
        elif header:
            current = header
        else:
            buf.append(line)
    if buf:
        sections.append((current, "\n".join(buf)))
    return [(s, t.strip()) for s, t in sections if t.strip()]


def window_chunks(text: str, section: str) -> list[tuple[str, str]]:
    """Sliding-window chunks of a section's text."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= CHUNK_CHARS:
        return [(section, text)] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        chunks.append((section, text[start:end]))
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def chunk_resume(row: pd.Series) -> list[dict]:
    """Chunk one resume row into embeddable documents with metadata."""
    meta = {
        "resume_id": str(row["id"]),
        "name": str(row["name"]),
        "designation": str(row["designation"]),
        "skills": str(row["skills"]),
        "companies": str(row["companies"]),
        "degree": str(row["degree"]),
        "college": str(row["college"]),
        "grad_year": str(row["grad_year"]),
        "years_experience": float(row["years_experience"])
        if str(row["years_experience"]) not in ("", "nan") else -1.0,
        "location": str(row["location"]),
    }
    docs = []
    for section, text in split_sections(str(row["text"])):
        for sec, chunk in window_chunks(text, section):
            docs.append({**meta, "section": sec, "text": chunk})
    if not docs:  # empty / unparseable resume: index the name line only
        docs.append({**meta, "section": "GENERAL",
                     "text": f"{row['name']} {row['designation']} {row['skills']}"})
    return docs


def build_chunks(df: pd.DataFrame | None = None) -> list[dict]:
    df = load_resumes() if df is None else df
    docs: list[dict] = []
    for _, row in df.iterrows():
        docs.extend(chunk_resume(row))
    return docs


def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def build_chroma(docs: list[dict] | None = None, rebuild: bool = False):
    """Build (or open) the persistent ChromaDB collection."""
    import chromadb

    if rebuild and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    if docs is not None and collection.count() == 0:
        embedder = get_embedder()
        texts = [d["text"] for d in docs]
        embs = embedder.encode(
            texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=True).tolist()
        collection.add(
            ids=[f"{d['resume_id']}-{i}" for i, d in enumerate(docs)],
            documents=texts,
            embeddings=embs,
            metadatas=[{k: v for k, v in d.items() if k != "text"} for d in docs],
        )
    return collection


def build_faiss_index(docs: list[dict] | None = None, path: Path | None = None):
    """Fallback vector store: FAISS flat inner-product index + JSON sidecar."""
    import json
    import faiss
    import numpy as np

    path = path or ROOT / "outputs" / "faiss_index"
    path.mkdir(parents=True, exist_ok=True)
    docs = build_chunks() if docs is None else docs
    embedder = get_embedder()
    embs = embedder.encode(
        [d["text"] for d in docs], batch_size=32,
        normalize_embeddings=True, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(path / "index.faiss"))
    (path / "chunks.json").write_text(
        json.dumps([{k: v for k, v in d.items()} for d in docs]),
        encoding="utf-8")
    print(f"FAISS index: {len(docs)} chunks, dim {embs.shape[1]} -> {path}")
    return index, docs
