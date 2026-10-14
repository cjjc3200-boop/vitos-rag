"""Embeddings — a tiny LangChain Embeddings implementation over fastembed.

LangChain's `Embeddings` interface is just two methods: `embed_documents` (for
the corpus) and `embed_query` (for the search question). By implementing it
ourselves we avoid the sunset `langchain-community` shim AND we can see exactly
what the interface requires.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding

from .config import EMBED_MODEL


class FastEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()
