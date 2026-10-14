"""Retrieve + generate — the RAG chain.

The system prompt enforces grounding: answer ONLY from context, admit when the
context does not contain the answer, and cite sources. This is what separates
RAG from a chatbot that makes things up.
"""
from __future__ import annotations

import os

from langchain_chroma import Chroma

from .config import CHROMA_DIR, COLLECTION, MIN_SCORE, TOP_K
from .embeddings import FastEmbeddings

SYSTEM_PROMPT = (
    "You are the customer assistant for Vito's Sandwich Shop. "
    "Answer using ONLY the context provided. "
    "If the answer is not in the context, say you don't know and suggest the "
    "customer call (813) 555-0142. "
    "Be concise and friendly. Cite the sources you used like [hours.md]."
)


def get_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=FastEmbeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(store: Chroma, question: str, k: int = TOP_K):
    """Return (docs, similarities).

    Chroma's default space is squared L2 distance. Because our embedding vectors
    are unit-length, that distance relates to cosine similarity by:

        d2 = 2 - 2*cos   ->   cos = 1 - d2 / 2

    So we recover a real cosine similarity in [-1, 1] (higher = more similar)
    and use that for both ranking display and the confidence guardrail.
    """
    results = store.similarity_search_with_score(question, k=k)
    docs = [d for d, _ in results]
    similarities = [1.0 - (d2 / 2.0) for _, d2 in results]
    return docs, similarities


def format_answer(question: str, docs, similarities) -> str:
    if not docs or (similarities and max(similarities) < MIN_SCORE):
        return (
            "I don't have that in Vito's information. "
            "Please call the shop at (813) 555-0142 and they'll help you."
        )

    lines = [f"Q: {question}", "", "Here's what I found:"]
    for d, sim in zip(docs, similarities):
        snippet = d.page_content.replace("\n", " ").strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + " …"
        lines.append(f"  • [{d.metadata.get('source', '?')}] ({sim:.2f}) {snippet}")
    return "\n".join(lines)


def answer_llm(question: str, docs) -> str:
    """Grounded answer via any OpenAI-compatible endpoint (OpenAI or Ollama)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("LLM mode needs langchain-openai:  pip install langchain-openai") from exc

    context = "\n\n".join(f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in docs)
    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    msgs = [("system", SYSTEM_PROMPT), ("user", f"Context:\n{context}\n\nQuestion: {question}")]
    return llm.invoke(msgs).content
