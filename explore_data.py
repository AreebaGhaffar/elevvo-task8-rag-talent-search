"""Explore the Resume Entities for NER dataset and build a clean resume table.

Input (data/, gitignored): traindata.json (200 resumes) + testdata.json
(20 resumes), DataTurks JSON export. Each item is
  {"content": "<full resume text>",
   "annotation": [{"label": ["Skills", ...],
                   "points": [{"start": int, "end": int, "text": "..."}]},
                  ...],
   "extras": null}

Output:
  results/exploration_summary.md  — counts, label distribution, text stats
  results/resumes.csv             — one row per resume: id, split, name,
    designation, degree, college, grad_year, years_experience, companies,
    skills, location, email, text (full resume text, truncated at 8000 chars)
  results/label_counts.csv        — entity counts per label

Usage:  python explore_data.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

LABEL_NAMES = [
    "Name", "College Name", "Degree", "Graduation Year",
    "Years of Experience", "Companies worked at", "Designation",
    "Skills", "Location", "Email Address",
]


def load_split(fname: str, split: str) -> list[dict]:
    path = DATA_DIR / fname
    # DataTurks export is JSONL: one JSON object per line.
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for i, item in enumerate(items):
        text = item.get("content", "")
        ents: dict[str, list[str]] = {}
        for ann in item.get("annotation") or []:
            labels = ann.get("label") or []
            for point in ann.get("points") or []:
                value = (point.get("text") or "").strip()
                if not value:
                    continue
                for label in labels:
                    ents.setdefault(label, []).append(value)
        # De-duplicate while preserving order.
        ents = {k: list(dict.fromkeys(v)) for k, v in ents.items()}
        rows.append({"id": f"{split}-{i:03d}", "split": split, "text": text, **ents})
    return rows


def first(values: list[str] | None) -> str:
    return (values or [""])[0]


def years_of_experience(values: list[str] | None) -> float | None:
    """Parse e.g. '5 Years' / '3.2-years' / '6 months' to a number of years.
    The data has annotation noise (two values are just a year, '2004');
    values above 50 are ignored as noise."""
    if not values:
        return None
    parsed = []
    for v in values:
        low = v.lower()
        m = re.search(r"(\d+(?:\.\d+)?)", low)
        if not m:
            continue
        num = float(m.group(1))
        if "month" in low:
            num /= 12
        if 0 < num <= 50:
            parsed.append(num)
    return max(parsed) if parsed else None


def main() -> None:
    train = load_split("traindata.json", "train")
    test = load_split("testdata.json", "test")
    rows = train + test
    print(f"train: {len(train)} resumes, test: {len(test)} resumes, total: {len(rows)}")

    label_counter: Counter = Counter()
    for r in rows:
        for label in LABEL_NAMES:
            if r.get(label):
                label_counter[label] += len(r[label])
    label_df = pd.DataFrame(
        [(label, label_counter.get(label, 0),
          sum(1 for r in rows if r.get(label)))
         for label in LABEL_NAMES],
        columns=["label", "entity_mentions", "resumes_with_label"],
    )
    label_df.to_csv(RESULTS / "label_counts.csv", index=False)
    print(label_df.to_string(index=False))

    # Flat resume table for the RAG pipeline.
    flat = []
    for r in rows:
        flat.append({
            "id": r["id"],
            "split": r["split"],
            "name": first(r.get("Name")),
            "designation": " | ".join(r.get("Designation") or []),
            "degree": " | ".join(r.get("Degree") or []),
            "college": " | ".join(r.get("College Name") or []),
            "grad_year": first(r.get("Graduation Year")),
            "years_experience": years_of_experience(r.get("Years of Experience")),
            "companies": " | ".join(r.get("Companies worked at") or []),
            "skills": " | ".join(r.get("Skills") or []),
            "location": " | ".join(r.get("Location") or []),
            "email": first(r.get("Email Address")),
            "text": r["text"][:8000],
        })
    df = pd.DataFrame(flat)
    df.to_csv(RESULTS / "resumes.csv", index=False)

    words = df["text"].str.split().str.len()
    n_skills = df["skills"].str.split(" | ", regex=False).apply(
        lambda x: len([s for s in x if s]))
    summary = f"""# Dataset exploration — Resume Entities for NER

## Source
- Kaggle (recommended): https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner
  - Owner DataTurks, 220 manually labelled resumes from indeed.com,
    10 entity categories, 1,220,776 bytes total, updated 2018-07-12.
- Files used (identical official files from the original publisher,
  DataTurks, because the Kaggle API download needs credentials and the
  Kaggle API host is unreachable through this VM's egress proxy):
  - `data/traindata.json` — 1,106,139 bytes, 200 resumes
  - `data/testdata.json` — 111,557 bytes, 20 resumes
  - https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy

## Size
- Total resumes: {len(df)} (train {len(train)}, test {len(test)})
- Resume text: mean {words.mean():.0f} words, median {words.median():.0f},
  min {words.min()}, max {words.max()}
- Resumes with a name: {(df['name'] != '').sum()}
- Resumes with a designation: {(df['designation'] != '').sum()}
- Resumes with skills: {(df['skills'] != '').sum()}
- Resumes with a degree: {(df['degree'] != '').sum()}
- Resumes with companies: {(df['companies'] != '').sum()}
- Mean Skills spans per resume: {n_skills.mean():.1f} (max {n_skills.max()}) — each span can list several skills
- Mean years of experience (where stated): {df['years_experience'].mean():.1f}
- Graduation years present: {df['grad_year'].replace('', pd.NA).dropna().nunique()} distinct
  values, range {df['grad_year'].replace('', pd.NA).dropna().min()}–
  {df['grad_year'].replace('', pd.NA).dropna().max()}

## Entity label counts (mentions across all resumes)
{label_df.to_string(index=False)}

## Notes for the RAG pipeline
- Only 220 resumes: the whole pool fits in one ChromaDB collection, so no
  sampling is needed for the index (unlike Tasks 6/7, where subsets were used).
- The `Skills` and `Designation` annotations are the gold labels used for
  retrieval evaluation (precision@k / recall@k in `src/evaluate.py`).
- No gender, ethnicity or age field exists in the data. The bias check
  therefore uses only explicitly labelled attributes present in the data
  (graduation-year cohort, college, location, years of experience) and
  never infers protected attributes from names.
- Data quirks: 2 annotations carry the label `UNKNOWN`; 2 'Years of
  Experience' values are just a year ('2004') and are treated as noise.
  Some resumes have no Skills annotation at all (37 of 220).
"""
    (RESULTS / "exploration_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
