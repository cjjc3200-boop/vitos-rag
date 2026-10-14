#!/usr/bin/env python3
"""Ask Vito's assistant a question.

Usage:
    python app.py                          # retrieval-only (no API key)
    python app.py --llm                    # grounded generated answer
    python app.py --once "do you have vegan options?"
"""
from __future__ import annotations

import argparse

from vitos.config import MIN_SCORE, TOP_K
from vitos.rag import answer_llm, format_answer, get_store, retrieve


def handle(store, question: str, use_llm: bool) -> str:
    docs, similarities = retrieve(store, question, k=TOP_K)
    if use_llm:
        if not docs or max(similarities) < MIN_SCORE:
            return format_answer(question, docs, similarities)
        return answer_llm(question, docs)
    return format_answer(question, docs, similarities)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vito's Sandwich Shop RAG assistant.")
    ap.add_argument("--llm", action="store_true", help="generate a grounded answer with an LLM")
    ap.add_argument("--once", metavar="QUESTION")
    args = ap.parse_args()

    store = get_store()

    if args.once:
        print(handle(store, args.once, args.llm))
        return

    print("Ask about Vito's: hours, menu, allergens, ordering, catering, payment.")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        print()
        print(handle(store, q, args.llm))
        print()


if __name__ == "__main__":
    main()
