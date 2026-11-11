"""Download the model checkpoints with curl.

Why curl, not the Hugging Face Hub downloader? On this VM the Hub
downloader stalls (Tasks 6/7 saw files come back as 0 bytes or truncated);
plain curl downloads the same files from the same Hub URLs at full speed.
Files land in models/<model-id>/ and are used directly, e.g.:

    SentenceTransformer("models/sentence-transformers/all-MiniLM-L6-v2")
    pipeline("text2text-generation", model="models/Qwen/Qwen2.5-0.5B-Instruct")

File lists were taken from the Hub API (https://huggingface.co/api/models/<id>)
on 2026-11-11. safetensors is preferred over pytorch_model.bin where both
exist (same weights, half the download).

Usage:  python download_models.py [model-id ...]
With no arguments, downloads all models used in the project.
"""
import subprocess
import sys
from pathlib import Path

BASE = "https://huggingface.co/{model}/resolve/main/{file}"
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": [
        "1_Pooling/config.json",
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
    "Qwen/Qwen2.5-0.5B-Instruct": [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
}


def expected_size(url: str) -> int | None:
    """Content-Length of a URL (following redirects), or None if unknown."""
    try:
        r = subprocess.run(
            ["curl", "-sIL", url], capture_output=True, text=True, timeout=60
        )
        for line in reversed(r.stdout.splitlines()):
            if line.lower().startswith("content-length:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:  # noqa: BLE001 - size check is best-effort
        pass
    return None


def download(model: str) -> None:
    dest = Path("models") / model
    dest.mkdir(parents=True, exist_ok=True)
    for fname in MODELS[model]:
        url = BASE.format(model=model, file=fname)
        target = dest / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        # Download with resume + size verification. The egress proxy sometimes
        # drops the connection near the end of a large file while curl still
        # exits 0, leaving a truncated file — so verify the size against the
        # server's Content-Length and retry (resuming) until they match.
        for attempt in range(1, 6):
            print(f"{model}/{fname} (attempt {attempt}) ...", flush=True)
            r = subprocess.run(
                ["curl", "-sS", "-L", "-C", "-", "-o", str(target), url]
            )
            if r.returncode != 0:
                print(f"  curl failed (exit {r.returncode})")
                continue
            expected = expected_size(url)
            got = target.stat().st_size if target.exists() else 0
            if expected is None or got == expected:
                print(f"  OK: {got:,} bytes")
                break
            print(f"  incomplete: got {got:,} of {expected:,} bytes — resuming")
        else:
            print(f"  FAILED after 5 attempts: {fname}")
    files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    print(f"  -> {dest}: {files}")
    weights = [f for f in files if f in ("model.safetensors", "pytorch_model.bin")]
    for w in weights:
        size = (dest / w).stat().st_size
        print(f"     {w}: {size:,} bytes {'OK' if size > 10_000_000 else 'SUSPICIOUSLY SMALL'}")


def main() -> None:
    models = sys.argv[1:] or list(MODELS)
    for model in models:
        if model not in MODELS:
            print(f"Unknown model: {model} (known: {sorted(MODELS)})")
            continue
        download(model)


if __name__ == "__main__":
    main()
