"""LLM-based evaluation: explain why the top-k candidates fit a job query.

Local LLM (default, no API key needed): Qwen/Qwen2.5-0.5B-Instruct
(causal LM, chat template, greedy decoding, CPU). It was chosen over
google/flan-t5-base after testing both on this machine:
- flan-t5-base echoed the query back as the "summary" and claimed
  candidates had skills they did not have.
- Qwen2.5-0.5B-Instruct writes fluent, grounded one-sentence profile
  summaries, but it has a strong affirmation bias: even when told a
  skill is missing, it still claims the candidate "excels in" it
  (tested 2026-11-11, see README "LLM limitations").

So the fit judgement itself is NOT left to free-form generation. It is
computed deterministically from the dataset's NER annotations and shown
next to the LLM's summary:
  1. Required skills are extracted from the query by matching it against
     the skill vocabulary of the resume pool.
  2. Per candidate, found/missing is decided by word-boundary matching
     against the candidate's annotated skills/designation/text.
  3. Verdict: STRONG (all required found), PARTIAL (some), WEAK (none).
  4. The LLM contributes a one-sentence profile summary (the part it
     does reliably) — never a fit claim on its own.

Optional API hook (for a reviewer with a key — no key is on file here):
set OPENAI_API_KEY or GEMINI_API_KEY and pass backend="openai"/"gemini".
The API path uses plain HTTPS (stdlib urllib) so no extra SDK is needed.
A stronger API model can be trusted with the free-form fit narrative;
the local path deliberately cannot.

Per-candidate Q&A ("Does this candidate have leadership experience?")
uses the same local model with the candidate's full resume text as
context, plus an extractive fallback: the resume sentence with the
highest keyword overlap with the question is always shown as evidence.
"""
import json
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
LOCAL_MODEL = "models/Qwen/Qwen2.5-0.5B-Instruct"
RESUMES_CSV = ROOT / "results" / "resumes.csv"

_LLM = None
_TOKENIZER = None


def get_local_llm():
    global _LLM, _TOKENIZER
    if _LLM is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        _LLM = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL, dtype=torch.float32)
    return _LLM, _TOKENIZER


def _chat(prompt: str, max_new_tokens: int = 100) -> str:
    model, tok = get_local_llm()
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def _truncate(text: str, words: int = 120) -> str:
    parts = text.split()
    return " ".join(parts[:words])


def skill_vocabulary() -> set[str]:
    """Skill phrases from the pool's annotated Skills.

    Every distinct phrase is kept (even one-off mentions): a skill like
    Tableau appears in only one resume's Skills annotation, and dropping
    it would make the query "…who knows SQL and Tableau" lose Tableau
    from the required-skills list. Junk phrases are harmless here because
    required_skills() only matches phrases that actually occur in the
    recruiter's query.
    """
    from collections import Counter
    df = pd.read_csv(RESUMES_CSV).fillna("")
    counter: Counter = Counter()
    for skills in df["skills"]:
        for span in str(skills).split(" | "):
            for part in re.split(r"[,;/]|\band\b", span):
                phrase = re.sub(r"\s*\([^)]*\)", "", part).strip().lower()
                phrase = re.sub(r"\s+", " ", phrase)
                if 1 < len(phrase) <= 30 and not phrase.isdigit():
                    counter[phrase] += 1
    return set(counter)


def required_skills(query: str, vocab: set[str] | None = None) -> list[str]:
    """Skills the query asks for, found by matching the query against the
    pool's skill vocabulary (longest phrases first, no subsumed matches)."""
    vocab = skill_vocabulary() if vocab is None else vocab
    q = query.lower()
    found = [v for v in vocab if re.search(rf"\b{re.escape(v)}\b", q)]
    found.sort(key=len, reverse=True)
    required = []
    for skill in found:
        if not any(re.search(rf"\b{re.escape(skill)}\b", other)
                   for other in required):
            required.append(skill)
    return sorted(required)


def _has_skill(candidate: dict, skill: str) -> bool:
    """Word-boundary match against the candidate's ANNOTATED skills and
    designation only (not the free-text evidence chunk: a passing mention
    of a tool in a resume is not the same as having it as a skill)."""
    hay = " ".join([str(candidate.get("skills", "")),
                    str(candidate.get("designation", ""))]).lower()
    return re.search(rf"\b{re.escape(skill)}\b", hay) is not None


def profile_summary(candidate: dict) -> str:
    """One-sentence LLM summary of the candidate's profile (grounded).
    The raw generation is cut to the first complete sentence so a
    rambling model cannot leave a half-finished sentence."""
    context = (
        f"Designation: {candidate.get('designation') or 'not stated'}. "
        f"Skills: {_truncate(candidate.get('skills') or 'not stated', 60)}. "
        f"Companies: {candidate.get('companies') or 'not stated'}. "
        f"Years of experience: {candidate.get('years_experience') if candidate.get('years_experience', -1) >= 0 else 'not stated'}. "
        f"Degree: {candidate.get('degree') or 'not stated'}."
    )
    raw = _chat(
        f"Summarize this candidate's experience in one sentence: {context}",
        max_new_tokens=120)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
    return sentences[0] if sentences else raw


def fit_summary_local(query: str, candidates: list[dict]) -> str:
    """Fit summary per candidate: verified skill match + LLM profile summary."""
    required = required_skills(query)
    out = []
    for c in candidates:
        found = [s for s in required if _has_skill(c, s)]
        missing = [s for s in required if s not in found]
        verdict = ("STRONG" if required and len(found) == len(required)
                   else "PARTIAL" if found else "WEAK")
        summary = profile_summary(c)
        out.append(
            f"**{c.get('name') or 'Unknown'}** — Fit: {verdict} "
            f"({len(found)}/{len(required)} required skills)\n"
            f"  LLM summary: {summary}\n"
            f"  Found: {', '.join(found) if found else 'none'} | "
            f"Missing: {', '.join(missing) if missing else 'none'}")
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
    if present, else local). The API backends let a stronger model write
    the free-form fit narrative; the local path uses the verified
    skill-match + LLM profile summary described above.
    """
    if backend == "auto":
        backend = ("openai" if os.environ.get("OPENAI_API_KEY")
                   else "gemini" if os.environ.get("GEMINI_API_KEY")
                   else "local")
    if backend in ("openai", "gemini"):
        chat = _openai_chat if backend == "openai" else _gemini_chat
        required = required_skills(query)
        blocks = []
        for c in candidates:
            found = [s for s in required if _has_skill(c, s)]
            missing = [s for s in required if s not in found]
            context = (
                f"Designation: {c.get('designation') or 'not stated'}. "
                f"Skills: {_truncate(c.get('skills') or 'not stated', 60)}. "
                f"Companies: {c.get('companies') or 'not stated'}. "
                f"Years of experience: {c.get('years_experience') if c.get('years_experience', -1) >= 0 else 'not stated'}. "
                f"Degree: {c.get('degree') or 'not stated'}."
            )
            prompt = (
                f"Job requirement: {query}\nCandidate: {c.get('name') or 'Unknown'}. "
                f"{context}\nVerified skill match — found: {found or 'none'}; "
                f"missing: {missing or 'none'}.\nIn two sentences, explain why "
                "this candidate fits the job requirement, using only the facts "
                "above. Do not claim a missing skill is present."
            )
            blocks.append(f"**{c.get('name') or 'Unknown'}** — {chat(prompt)}")
        return "\n\n".join(blocks) + f"\n\n_(LLM backend: {backend})_"
    return (fit_summary_local(query, candidates)
            + "\n\n_(LLM backend: local Qwen2.5-0.5B-Instruct; fit verdicts are "
              "computed from the annotated skills, not generated — see README)_")


def answer_about_candidate(resume_text: str, question: str) -> dict:
    """Answer a question about one candidate's resume.

    Returns {"answer": ..., "evidence": <best matching resume sentence>,
    "method": "llm" | "extractive"}. The LLM answer is generated with the
    resume as context; the evidence sentence is the extractive fallback
    and is always shown so the answer can be checked.
    """
    context = _truncate(resume_text, 350)
    try:
        answer = _chat(
            f"Resume:\n{context}\n\nQuestion: {question}\n"
            "Answer the question using only the resume above. "
            "If the resume does not say, answer 'Not stated in this resume.'",
            max_new_tokens=80)
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
