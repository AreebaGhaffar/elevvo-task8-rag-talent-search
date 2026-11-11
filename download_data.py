"""Download the Resume Entities for NER dataset for Task 8.

Recommended source (per the task): Kaggle
  https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner
  ("Resume Entities for NER", owner DataTurks, 220 manually labelled
  resumes from indeed.com, 10 entity categories, 1,220,776 bytes total,
  updated 2018-07-12, licence Unknown — verified via the Kaggle API
  view endpoint on 2026-11-11.)

The Kaggle download needs Kaggle credentials, and the Kaggle API host is
unreachable through this VM's egress proxy for authenticated downloads
(same as Tasks 6/7), so this script falls back to the IDENTICAL official
files from their original publisher, DataTurks, on GitHub:
  https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy
  traindata.json  (1,106,139 bytes, 200 resumes)
  testdata.json   (  111,557 bytes,  20 resumes)
The Kaggle dataset is a copy of this DataTurks export (the Kaggle page
itself links the DataTurks project page, and the counts match exactly:
200 + 20 = 220 items). Files are downloaded with curl and verified
against the expected byte sizes.

What is saved (all in data/, gitignored):
  traindata.json  — 200 annotated resumes (DataTurks JSON export)
  testdata.json   —  20 annotated resumes

Usage:  python download_data.py
"""
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Original publisher files (identical to the Kaggle dataset contents).
FILES = {
    "traindata.json": (
        "https://raw.githubusercontent.com/DataTurks-Engg/"
        "Entity-Recognition-In-Resumes-SpaCy/master/traindata.json",
        1_106_139,
    ),
    "testdata.json": (
        "https://raw.githubusercontent.com/DataTurks-Engg/"
        "Entity-Recognition-In-Resumes-SpaCy/master/testdata.json",
        111_557,
    ),
}


def try_kagglehub() -> bool:
    """Attempt the Kaggle route first (works only if Kaggle credentials
    are configured on this machine). Returns True on success."""
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed — skipping Kaggle route.")
        return False
    try:
        path = kagglehub.dataset_download("dataturks/resume-entities-for-ner")
        print(f"Kaggle download succeeded: {path}")
        return True
    except Exception as e:  # noqa: BLE001 - record and fall back
        print(f"Kaggle download failed ({e}) — falling back to DataTurks GitHub.")
        return False


def curl(url: str, dest: Path, expected: int) -> bool:
    print(f"Downloading {url} -> {dest} ...", flush=True)
    r = subprocess.run(["curl", "-sS", "-L", "-C", "-", "-o", str(dest), url])
    ok = (
        r.returncode == 0
        and dest.exists()
        and dest.stat().st_size == expected
    )
    size = dest.stat().st_size if dest.exists() else 0
    print(f"  {'OK' if ok else 'FAILED'}: {size:,} bytes (expected {expected:,})")
    return ok


def main() -> None:
    ok = True
    if not try_kagglehub():
        for name, (url, expected) in FILES.items():
            ok &= curl(url, DATA_DIR / name, expected)
    if not ok:
        print("ERROR: one or more files failed verification.")
        sys.exit(1)
    print("Dataset download complete.")


if __name__ == "__main__":
    main()
