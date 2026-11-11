"""Semantic search over the resume vector store.

Query -> embedding -> ChromaDB top-k chunks -> aggregate per resume by
the best chunk score -> top-k resumes with the matching chunk as evidence.
Scores are cosine similarities in [0, 1] (ChromaDB returns distances;
we convert with score = 1 - distance for the cosine space).
"""
from pathlib import Path

from . import resume_store

ROOT = Path(__file__).parent.parent


def search(query: str, k: int = 5, n_chunks: int = 20,
           collection=None) -> list[dict]:
    """Return the top-k resumes for a natural-language query."""
    embedder = resume_store.get_embedder()
    q_emb = embedder.encode(
        [query], normalize_embeddings=True).tolist()
    collection = collection or resume_store.build_chroma()
    res = collection.query(
        query_embeddings=q_emb, n_results=min(n_chunks, collection.count()))
    # Aggregate chunk hits per resume (best chunk score wins).
    best: dict[str, dict] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0]):
        rid = meta["resume_id"]
        score = round(1.0 - float(dist), 4)
        if rid not in best or score > best[rid]["score"]:
            best[rid] = {
                "resume_id": rid,
                "name": meta.get("name", ""),
                "designation": meta.get("designation", ""),
                "skills": meta.get("skills", ""),
                "companies": meta.get("companies", ""),
                "degree": meta.get("degree", ""),
                "college": meta.get("college", ""),
                "grad_year": meta.get("grad_year", ""),
                "years_experience": meta.get("years_experience", -1),
                "location": meta.get("location", ""),
                "score": score,
                "evidence_section": meta.get("section", ""),
                "evidence": doc,
            }
    ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:k]


def format_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r['name'] or '(no name)'} — {r['designation'] or 'no designation'} "
            f"(score {r['score']:.3f})")
        if r["skills"]:
            lines.append(f"   Skills: {r['skills'][:200]}")
        if r["companies"]:
            lines.append(f"   Companies: {r['companies'][:150]}")
        lines.append(f"   Evidence [{r['evidence_section']}]: "
                     f"{r['evidence'][:220].replace(chr(10), ' ')}")
    return "\n".join(lines)
