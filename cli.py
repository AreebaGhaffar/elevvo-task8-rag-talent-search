"""Command-line interface for the RAG talent search engine.

Usage:
  python cli.py build [--rebuild] [--faiss]
      Build the ChromaDB index (and optionally the FAISS fallback).
  python cli.py search "Junior Data Analyst who knows SQL and Tableau" [--k 5]
      Semantic search; prints ranked candidates with scores + evidence.
  python cli.py explain "Junior Data Analyst who knows SQL and Tableau" [--k 3]
      Search + LLM fit summary for the top-k (local flan-t5-base by default;
      set OPENAI_API_KEY / GEMINI_API_KEY and --backend openai|gemini).
  python cli.py ask <resume-id> "Does this candidate have leadership experience?"
      Answer a question about one candidate's resume.
  python cli.py bias "Junior Data Analyst who knows SQL and Tableau" [--k 10]
      Bias check on the retrieved top-k vs the whole pool.
  python cli.py evaluate
      Precision@5 / @10 and recall@10 on 10 hand-built queries.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src import bias as bias_mod
from src import llm as llm_mod
from src import resume_store
from src.search import format_results, search


def cmd_build(args):
    import subprocess
    cmd = [sys.executable, "build_index.py"]
    if args.rebuild:
        cmd.append("--rebuild")
    if args.faiss:
        cmd.append("--faiss")
    raise SystemExit(subprocess.call(cmd))


def cmd_search(args):
    results = search(args.query, k=args.k)
    print(f"Query: {args.query}\n")
    print(format_results(results))
    return results


def cmd_explain(args):
    results = search(args.query, k=args.k)
    print(f"Query: {args.query}\n")
    print(format_results(results))
    print("\n--- LLM fit summary ---\n")
    print(llm_mod.fit_summary(args.query, results, backend=args.backend))


def cmd_ask(args):
    df = resume_store.load_resumes()
    row = df[df["id"] == args.resume_id]
    if row.empty:
        print(f"No resume with id {args.resume_id}")
        return
    text = str(row.iloc[0]["text"])
    name = row.iloc[0]["name"]
    out = llm_mod.answer_about_candidate(text, args.question)
    print(f"Candidate: {name} ({args.resume_id})")
    print(f"Q: {args.question}")
    print(f"A: {out['answer']}  (method: {out['method']})")
    if out["evidence"]:
        print(f"Evidence: {out['evidence'][:300]}")


def cmd_bias(args):
    results = search(args.query, k=args.k)
    df = resume_store.load_resumes()
    report = bias_mod.bias_check([r["resume_id"] for r in results], df)
    print(f"Query: {args.query}\n")
    print(format_results(results))
    print()
    print(bias_mod.format_report(report))


def cmd_evaluate(_args):
    from src import evaluate
    evaluate.main()


def main():
    ap = argparse.ArgumentParser(description="RAG talent search engine")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--faiss", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("explain")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--backend", default="auto",
                   choices=["auto", "local", "openai", "gemini"])
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("ask")
    p.add_argument("resume_id")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("bias")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=10)
    p.set_defaults(func=cmd_bias)

    p = sub.add_parser("evaluate")
    p.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
