"""Ingest: load data/ -> split into chunks -> embed -> store in Chroma.

Re-running this rebuilds the collection from scratch so it is always in sync
with data/.

Usage:
    python -m vitos.ingest
"""
from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION
from .embeddings import FastEmbeddings
from .loader import load_documents


def build_store() -> Chroma:
    docs = load_documents()
    if not docs:
        raise SystemExit("No documents found in data/. Add .md files there.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", " "],   # prefer breaking on headings
    )
    chunks = splitter.split_documents(docs)
    print(f"  loaded {len(docs)} document(s)")
    print(f"  split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)          # rebuild cleanly each time

    store = Chroma.from_documents(
        documents=chunks,
        embedding=FastEmbeddings(),
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  embedded + stored in Chroma at {CHROMA_DIR}")
    return store


if __name__ == "__main__":
    print("Ingesting Vito's knowledge base…")
    build_store()
    print("Done. Now run:  python app.py")
