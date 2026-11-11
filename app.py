"""Streamlit UI: chat with the resume database.

Recruiter flow: type a natural-language query -> see ranked candidates
with similarity scores and evidence -> read the LLM fit summary ->
check the bias panel -> ask follow-up questions about any candidate
("Does this candidate have leadership experience?").

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src import bias as bias_mod
from src import llm as llm_mod
from src import resume_store
from src.search import search

st.set_page_config(page_title="Talent Search — RAG", layout="wide")
st.title("Talent Search Engine (RAG)")
st.caption("Semantic search over 220 resumes · ChromaDB + all-MiniLM-L6-v2 · "
           "LLM fit summaries from local flan-t5-base (no API key needed)")


@st.cache_resource
def load_pool():
    return resume_store.load_resumes()


@st.cache_resource
def load_collection():
    return resume_store.build_chroma()


pool = load_pool()
collection = load_collection()
st.sidebar.markdown(f"**Index:** {collection.count()} chunks, "
                    f"{len(pool)} resumes")
st.sidebar.markdown("**Embedding:** all-MiniLM-L6-v2 (384-dim, cosine)")
st.sidebar.markdown("**LLM:** flan-t5-base (local) — set OPENAI_API_KEY / "
                    "GEMINI_API_KEY to swap in an API model")
k = st.sidebar.slider("Candidates to show", 3, 10, 5)

query = st.text_input(
    "Ask for a candidate",
    placeholder="Find me a Junior Data Analyst who knows SQL and Tableau")

if query:
    with st.spinner("Searching the resume database..."):
        results = search(query, k=k, collection=collection)
    st.subheader(f"Top {len(results)} candidates")
    for r in results:
        with st.expander(
                f"{r['name'] or '(no name)'} — {r['designation'] or 'no designation'} "
                f"(score {r['score']:.3f})"):
            st.markdown(f"**Skills:** {r['skills'] or 'not stated'}")
            st.markdown(f"**Companies:** {r['companies'] or 'not stated'}")
            st.markdown(f"**Degree:** {r['degree'] or 'not stated'} · "
                        f"**Experience:** "
                        f"{r['years_experience'] if r['years_experience'] >= 0 else 'not stated'} yrs · "
                        f"**Location:** {r['location'] or 'not stated'}")
            st.markdown(f"**Evidence [{r['evidence_section']}]:** {r['evidence'][:500]}")
            q = st.text_input(
                "Ask about this candidate",
                placeholder="Does this candidate have leadership experience?",
                key=f"ask-{r['resume_id']}")
            if q:
                row = pool[pool["id"] == r["resume_id"]]
                if not row.empty:
                    out = llm_mod.answer_about_candidate(
                        str(row.iloc[0]["text"]), q)
                    st.markdown(f"**Answer ({out['method']}):** {out['answer']}")
                    if out["evidence"]:
                        st.markdown(f"*Evidence: {out['evidence'][:300]}*")

    if st.button("Generate LLM fit summary (top 3)"):
        with st.spinner("Asking the local LLM..."):
            st.markdown(llm_mod.fit_summary(query, results[:3]))

    with st.expander("Bias check — does the top-k favour one group?"):
        report = bias_mod.bias_check([r["resume_id"] for r in results], pool)
        st.markdown("```\n" + bias_mod.format_report(report) + "\n```")
