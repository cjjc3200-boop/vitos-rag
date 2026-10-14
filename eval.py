#!/usr/bin/env python3
"""A small evaluation harness.

This is the thing that separates a portfolio project from a toy: a fixed set of
questions with known correct source files, scored automatically. It is a
miniature 'LLM-as-judge / retrieval-accuracy' eval.

Metric: Hit rate — did the correct source appear in the retrieved chunks,
while irrelevant questions correctly trigger the 'I don't know' guard?

Run:  python eval.py
"""
from __future__ import annotations

from vitos.config import MIN_SCORE, TOP_K
from vitos.rag import get_store, retrieve

# (question, expected source file)
CASES: list[tuple[str, str | None]] = [
    ("What time do you close on Saturday?", "hours.md"),
    ("Are you open on Sunday?", "hours.md"),
    ("How much is The Vito sandwich?", "menu.md"),
    ("Do you have gluten free bread?", "allergens.md"),
    ("Which sandwiches are vegetarian?", "allergens.md"),
    ("Can I order delivery?", "ordering.md"),
    ("How much notice for a catering order?", "catering.md"),
    ("Do you take Apple Pay?", "payment.md"),
    ("How does the loyalty punch card work?", "payment.md"),
    ("What is the capital of France?", None),   # should trigger the guard
]


def main() -> None:
    store = get_store()
    hits = 0

    print(f"Evaluating {len(CASES)} questions (top_k={TOP_K}, min_score={MIN_SCORE})\n")
    for question, expected in CASES:
        docs, similarities = retrieve(store, question)
        sources = {d.metadata.get("source") for d in docs}
        confident = bool(docs) and max(similarities) >= MIN_SCORE

        if expected is None:
            ok = not confident
            label = "guard" if ok else "ANSWERED (should have refused)"
        else:
            ok = confident and expected in sources
            label = f"src={expected}" if ok else f"FAIL (got {sorted(s for s in sources if s)})"

        if ok:
            hits += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {question}  ->  {label}")

    n = len(CASES)
    print(f"\nOverall accuracy: {hits}/{n} = {hits / n:.0%}")


if __name__ == "__main__":
    main()
