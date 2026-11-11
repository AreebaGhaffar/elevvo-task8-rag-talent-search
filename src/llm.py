"""LLM-based evaluation: explain why the top-k candidates fit a job query.

Local LLM (default, no API key needed): google/flan-t5-base
(text2text-generation, CPU). It is small enough for this machine and
produces short, grounded fit summaries when given the candidate's
designation, skills and experience as context.

Optional API hook (for a reviewer with a key — no key is on file here):
set OPENAI_API_KEY or GEMINI_API_KEY and pass backend="openai"/"gemini".
The API path uses plain HTTPS (stdlib urllib) so no extra SDK is needed.
If no key is set, the API backends raise a clear error and the caller
falls back to the local model.

Per-candidate Q&A ("Does this candidate have leadership experience?")
uses the same local model with the candidate's full resume text as
context, plus an extractive fallback: the resume sentence with the
highest keyword overlap with the question.
"""
import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOCAL_MODEL = "models/google/flan-t5-base"

_LLM = None


def get_local_llm():
    global _LLM
    if _LLM is None:
        from transformers import pipeline
        _LLM = pipeline(
            "text2text-generation", model=LOCAL_MODEL,
            max_new_tokens=160, do_sample=False, device=-1)
    return _LLM


def _truncate(text: str, words: int = 120) -> str:
    parts = text.split()
    return " ".join(parts[:words])


def fit_summary_local(query: str, candidates: list[dict]) -> str:
    """One short fit paragraph per candidate, from the local LLM."""
    llm = get_local_llm()
    out = []
    for c in candidates:
        context = (
            f"Designation: {c.get('designation') or 'not stated'}. "
            f"Skills: {_truncate(c.get('skills') or 'not stated', 60)}. "
            f"Companies: {c.get('companies') or 'not stated'}. "
            f"Years of experience: {c.get('years_experience') if c.get('years_experience', -1) >= 0 else 'not stated'}. "
            f"Degree: {c.get('degree') or 'not stated'}."
        )
        prompt = (
            f"Job requirement: {query}\n"
            f"Candidate: {c.get('name') or 'Unknown'}. {context}\n"
            "In two sentences, explain why this candidate fits the job "
            "requirement, using only the facts above. If a required skill "
            "is missing, say so."
        )
        gen = llm(prompt)[0]["generated_text"]
        out.append(f"**{c.get('name') or 'Unknown'}** — {gen.strip()}")
    return "\n\n".join(out)


def _openai_chat(prompt: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set — cannot use the OpenAI backend.")
    body = json.dumps({
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def _gemini_chat(prompt: str) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set — cannot use the Gemini backend.")
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:"
           f"generateContent?key={key}")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def fit_summary(query: str, candidates: list[dict], backend: str = "auto") -> str:
    """LLM fit summary for the top-k candidates.

    backend: "local" (default), "openai", "gemini", or "auto" (API key
    if present, else local). Returns the summary plus a one-line note
    naming the backend actually used.
    """
    if backend == "auto":
        backend = ("openai" if os.environ.get("OPENAI_API_KEY")
                   else "gemini" if os.environ.get("GEMINI_API_KEY")
                   else "local")
    if backend in ("openai", "gemini"):
        chat = _openai_chat if backend == "openai" else _gemini_chat
        blocks = []
        for c in candidates:
            context = (
                f"Designation: {c.get('designation') or 'not stated'}. "
                f"Skills: {_truncate(c.get('skills') or 'not stated', 60)}. "
                f"Companies: {c.get('companies') or 'not stated'}. "
                f"Years of experience: {c.get('years_experience') if c.get('years_experience', -1) >= 0 else 'not stated'}. "
                f"Degree: {c.get('degree') or 'not stated'}."
            )
            prompt = (
                f"Job requirement: {query}\nCandidate: {c.get('name') or 'Unknown'}. "
                f"{context}\nIn two sentences, explain why this candidate fits "
                "the job requirement, using only the facts above. If a required "
                "skill is missing, say so."
            )
            blocks.append(f"**{c.get('name') or 'Unknown'}** — {chat(prompt)}")
        return "\n\n".join(blocks) + f"\n\n_(LLM backend: {backend})_"
    return fit_summary_local(query, candidates) + "\n\n_(LLM backend: local flan-t5-base)_"


def answer_about_candidate(resume_text: str, question: str) -> dict:
    """Answer a question about one candidate's resume.

    Returns {"answer": ..., "evidence": <best matching resume sentence>,
    "method": "llm" | "extractive"}. The LLM answer is generated with the
    resume as context; the evidence sentence is the extractive fallback
    and is always shown so the answer can be checked.
    """
    llm = get_local_llm()
    context = _truncate(resume_text, 350)
    prompt = (f"Resume:\n{context}\n\nQuestion: {question}\n"
              "Answer the question using only the resume above. "
              "If the resume does not say, answer 'Not stated in this resume.'")
    try:
        answer = llm(prompt)[0]["generated_text"].strip()
        method = "llm"
    except Exception:  # noqa: BLE001 - fall back to extractive
        answer, method = "", "extractive"
    # Extractive evidence: sentence with max keyword overlap.
    words = set(re.findall(r"[a-z0-9]+", question.lower())) - {
        "does", "this", "candidate", "have", "the", "a", "an", "is", "are",
        "what", "which", "did", "do", "any", "with", "in", "of", "to"}
    best, best_score = "", -1
    for sent in re.split(r"(?<=[.!?])\s+", resume_text.replace("\n", " ")):
        sent_words = set(re.findall(r"[a-z0-9]+", sent.lower()))
        score = len(words & sent_words)
        if score > best_score:
            best, best_score = sent.strip(), score
    if not answer:
        answer = best if best_score > 0 else "Not stated in this resume."
        method = "extractive"
    return {"answer": answer, "evidence": best, "method": method}
