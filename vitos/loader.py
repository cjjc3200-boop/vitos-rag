"""Load the shop's knowledge base from data/*.md into LangChain Documents.

Keeping each file as its own Document preserves the source filename as
metadata, which is what lets the assistant cite where an answer came from.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from .config import DATA_DIR


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs
