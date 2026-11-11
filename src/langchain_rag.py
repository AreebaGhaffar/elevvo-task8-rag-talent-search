"""LangChain integration for the same RAG pipeline.

The core pipeline (src/search.py + src/llm.py) is implemented directly
on ChromaDB + sentence-transformers so it has no LangChain dependency.
This module shows the equivalent LangChain wiring, using the same local
models and the same ChromaDB collection:

  - langchain_chroma.Chroma + langchain_huggingface.HuggingFaceEmbeddings
    (local all-MiniLM-L6-v2) for retrieval
  - langchain_huggingface.HuggingFacePipeline (local flan-t5-base) for
    the LLM fit summary, via a RetrievalQA-style chain

Usage:  python -m src.langchain_rag "Junior Data Analyst who knows SQL and Tableau"
"""
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline as hf_pipeline

from . import resume_store

EMBED_MODEL = resume_store.EMBED_MODEL
LLM_MODEL = "models/google/flan-t5-base"


def build_chain(k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        collection_name=resume_store.COLLECTION_NAME,
        persist_directory=str(resume_store.CHROMA_DIR),
        embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = HuggingFacePipeline(pipeline=hf_pipeline(
        "text2text-generation", model=LLM_MODEL,
        max_new_tokens=160, do_sample=False, device=-1))
    prompt = PromptTemplate.from_template(
        "Job requirement: {question}\n\nRetrieved candidate excerpts:\n{context}\n\n"
        "In two sentences per candidate, explain why each candidate fits "
        "the job requirement, using only the facts above.")
    return retriever, llm, prompt


def ask(question: str, k: int = 3) -> str:
    retriever, llm, prompt = build_chain(k)
    docs = retriever.invoke(question)
    context = "\n---\n".join(d.page_content for d in docs)
    out = llm.invoke(prompt.format(question=question, context=context))
    return out.strip()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "Junior Data Analyst who knows SQL and Tableau")
    print(ask(question))
