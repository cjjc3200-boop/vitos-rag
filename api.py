#!/usr/bin/env python3
"""Vito's Sandwich Shop — HTTP API.

Turns the RAG pipeline into a live service. This is the difference between a
repo that runs in a terminal and a product you can point a URL at.

Endpoints:
    GET  /health   -> liveness + whether the index is loaded
    POST /ask      -> { "question": "..." } -> grounded answer + sources + trace

Run locally:
    uvicorn api:app --reload --port 8000
    # then: curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
    #            -d '{"question":"do you have gluten free bread?"}'
Run with Docker:
    docker build -t vitos-api . && docker run -p 8000:8000 vitos-api
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from vitos.config import MIN_SCORE, TOP_K
from vitos.rag import answer_llm, format_answer, get_store, retrieve
from vitos.tracing import Trace, langsmith_enabled, log_trace, span

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")

USE_LLM = os.environ.get("VITOS_LLM", "false").lower() == "true"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the vector store once at startup, not per request.
    state["store"] = get_store()
    yield
    state.clear()


app = FastAPI(title="Vito's Sandwich Shop — RAG API", version="0.2.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class AskResponse(BaseModel):
    question: str
    answer: str
    answered: bool
    sources: list[str]
    top_score: float | None
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_loaded": "store" in state,
        "llm_mode": USE_LLM,
        "langsmith_tracing": langsmith_enabled(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    store = state["store"]
    trace = Trace(question=req.question)

    with span(trace, "retrieve", top_k=TOP_K):
        docs, similarities = retrieve(store, req.question, k=TOP_K)

    trace.sources = [d.metadata.get("source", "?") for d in docs]
    trace.top_score = round(max(similarities), 4) if similarities else None
    confident = bool(docs) and trace.top_score is not None and trace.top_score >= MIN_SCORE
    trace.answered = confident

    with span(trace, "generate", mode="llm" if (USE_LLM and confident) else "retrieval"):
        if USE_LLM and confident:
            answer = answer_llm(req.question, docs)
        else:
            answer = format_answer(req.question, docs, similarities)

    log_trace(trace)

    return AskResponse(
        question=req.question,
        answer=answer,
        answered=confident,
        sources=trace.sources,
        top_score=trace.top_score,
        latency_ms=round(sum(s.duration_ms for s in trace.spans), 2),
    )
