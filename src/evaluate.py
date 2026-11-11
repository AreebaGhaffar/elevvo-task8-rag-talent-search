"""Retrieval evaluation: precision@k / recall@k on hand-built queries.

Gold sets are derived from the dataset's own NER annotations, so the
evaluation is reproducible without extra labelling:
- A resume is GOLD for a query if its annotated Skills contain at least
  one required skill (case-insensitive substring) AND its Designation
  (or Skills text) mentions the role keyword, when the query names one.
- Queries were written to cover distinct roles/skills in the data
  (analyst, developer, tester, designer, support, sales, ...).

Metrics (all on the same fixed pool of 220 resumes):
  P@5  = relevant in top-5 / 5
  P@10 = relevant in top-10 / 10
  R@10 = relevant in top-10 / all gold in the pool

Usage:  python -m src.evaluate [--k 10]
Writes results/retrieval_evaluation.json and .csv.
"""
import json
from pathlib import Path

import pandas as pd

from . import resume_store
from .search import search

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

# query, required skills (any one counts), role keywords (any one counts)
QUERIES = [
    ("Junior Data Analyst who knows SQL and Tableau",
     ["sql", "tableau"], ["analyst", "data"]),
    ("Java developer with Spring and Oracle experience",
     ["java", "spring", "oracle"], ["developer", "software", "java"]),
    ("Software tester who knows Selenium and manual testing",
     ["selenium", "testing"], ["tester", "test", "qa"]),
    ("Web developer with HTML, CSS and JavaScript",
     ["html", "css", "javascript"], ["developer", "web"]),
    ("Network engineer with CCNA and routing",
     ["ccna", "network", "routing"], ["network", "engineer"]),
    ("Customer support executive with good communication",
     ["communication"], ["support", "customer", "executive"]),
    ("UI designer with Photoshop and user experience",
     ["photoshop", "ui", "ux"], ["designer", "ui"]),
    ("Accountant who knows Tally and Excel",
     ["tally", "excel"], ["accountant", "account"]),
    ("Project manager with Agile and leadership",
     ["agile", "leadership"], ["manager", "project"]),
    ("Data entry operator with typing and MS Office",
     ["ms office", "typing", "data entry"], ["data entry", "operator"]),
]


def gold_ids(query_skills: list[str], role_keywords: list[str],
             df: pd.DataFrame) -> set[str]:
    gold = set()
    for _, row in df.iterrows():
        skills = str(row["skills"]).lower()
        desig = str(row["designation"]).lower()
        has_skill = any(s in skills for s in query_skills)
        has_role = (not role_keywords
                    or any(k in desig or k in skills for k in role_keywords))
        if has_skill and has_role:
            gold.add(str(row["id"]))
    return gold


def main() -> None:
    df = resume_store.load_resumes()
    rows = []
    for query, skills, roles in QUERIES:
        gold = gold_ids(skills, roles, df)
        retrieved = search(query, k=10, n_chunks=30)
        ret_ids = [r["resume_id"] for r in retrieved]
        rel5 = sum(1 for r in ret_ids[:5] if r in gold)
        rel10 = sum(1 for r in ret_ids if r in gold)
        rows.append({
            "query": query,
            "n_gold_in_pool": len(gold),
            "precision_at_5": round(rel5 / 5, 3),
            "precision_at_10": round(rel10 / 10, 3),
            "recall_at_10": round(rel10 / len(gold), 3) if gold else None,
            "retrieved_top5": "; ".join(ret_ids[:5]),
        })
        print(f"{query[:55]:<55} gold={len(gold):>3}  P@5={rel5/5:.2f}  "
              f"P@10={rel10/10:.2f}  R@10={rel10/len(gold) if gold else 0:.2f}")
    eval_df = pd.DataFrame(rows)
    summary = {
        "n_queries": len(rows),
        "pool_size": len(df),
        "mean_precision_at_5": round(eval_df["precision_at_5"].mean(), 3),
        "mean_precision_at_10": round(eval_df["precision_at_10"].mean(), 3),
        "mean_recall_at_10": round(eval_df["recall_at_10"].mean(), 3),
        "queries": rows,
    }
    (RESULTS / "retrieval_evaluation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    eval_df.to_csv(RESULTS / "retrieval_evaluation.csv", index=False)
    print(f"\nMean P@5={summary['mean_precision_at_5']}  "
          f"P@10={summary['mean_precision_at_10']}  "
          f"R@10={summary['mean_recall_at_10']}  (n={len(rows)} queries, "
          f"pool={len(df)} resumes)")


if __name__ == "__main__":
    main()
