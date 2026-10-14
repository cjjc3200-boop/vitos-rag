"""Vito's Sandwich Shop — RAG assistant.

A portfolio project demonstrating the applied-AI stack job postings name:
LangChain + a vector database (Chroma) + RAG, with an eval harness.

Stages:
    load     -> loader.py
    chunk    -> splitter in ingest.py
    embed    -> embeddings.py
    store    -> Chroma (persistent)
    retrieve -> rag.py
    generate -> rag.py (grounded, with citations)
    evaluate -> eval.py
"""

__version__ = "0.1.0"
