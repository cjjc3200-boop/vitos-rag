"""Lightweight tracing for the RAG pipeline.

Observability is one of the things that separates a hireable candidate from a
hobbyist. Full LangSmith tracing is great, but you should understand what a
trace even *is*: a record of each stage of a request — how long it took, what
it retrieved, and what it cost.

This module does two things:
  1. A tiny in-process tracer that records a structured span per request and
     logs it as JSON. Zero external services.
  2. An optional LangSmith hook: if LANGCHAIN_TRACING_V2=true and
     LANGCHAIN_API_KEY are set, LangChain will emit full traces automatically
     (we just detect it and report that it's on).
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger("vitos.trace")


@dataclass
class Span:
    name: str
    duration_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    question: str
    spans: list[Span] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    top_score: float | None = None
    answered: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "question": self.question,
                "answered": self.answered,
                "top_score": self.top_score,
                "sources": self.sources,
                "total_ms": round(sum(s.duration_ms for s in self.spans), 2),
                "spans": [asdict(s) for s in self.spans],
            },
            ensure_ascii=False,
        )


@contextmanager
def span(trace: Trace, name: str, **attributes: Any):
    """Time a block and attach it to the trace."""
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000
        trace.spans.append(Span(name=name, duration_ms=ms, attributes=attributes))


def langsmith_enabled() -> bool:
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true" and bool(
        os.environ.get("LANGCHAIN_API_KEY")
    )


def log_trace(trace: Trace) -> None:
    log.info("trace %s", trace.to_json())
