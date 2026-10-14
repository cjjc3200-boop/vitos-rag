# Vito's Sandwich Shop — RAG Customer Assistant

A production-shaped Retrieval-Augmented Generation (RAG) assistant for a
restaurant, built with the applied-AI stack that job postings actually name:
**LangChain + Chroma (vector database) + RAG**, plus a real evaluation harness.

The "client" is a mock Tampa Bay sandwich shop. The point of the project is to
demonstrate the whole pipeline *and* the things that separate a hireable
candidate from a hobbyist: **grounding, citations, a confidence guardrail, and
automated evals.**

---

## What it does

Ask it anything a customer would ask Vito's — hours, menu, allergens, ordering,
catering, payment — and it answers from the shop's own documents, citing the
source file. Ask something off-topic and it refuses instead of making
something up.

```
you> do you have gluten free bread?

Q: do you have gluten free bread?

Here's what I found:
  • [allergens.md] (0.71) Gluten — present in all standard rolls, breaded items
    (eggplant parm, chicken cutlet), and pasta. Gluten-free rolls are
    available for +$2. …
```

```
you> what is the capital of France?

I don't have that in Vito's information. Please call the shop at
(813) 555-0142 and they'll help you.
```

---

## The pipeline (and where each stage lives)

| Stage | Concept | File |
|-------|---------|------|
| Load | Read the knowledge base as Documents with source metadata | `vitos/loader.py` |
| Chunk | Split into overlapping passages, preferring heading breaks | `vitos/ingest.py` |
| Embed | Text → vectors (local, no API key) | `vitos/embeddings.py` |
| Store | Persist vectors in Chroma | `vitos/ingest.py` |
| Retrieve | Similarity search, top-k | `vitos/rag.py` |
| Generate | Grounded answer with citations | `vitos/rag.py` |
| **Guardrail** | Refuse when nothing matches well enough | `vitos/config.py` (`MIN_SCORE`) |
| **Evaluate** | Score retrieval against a fixed test set | `eval.py` |

### The deliberate design choices

- **A hand-written LangChain `Embeddings` class** (`embeddings.py`). Rather than
  depend on the sunset `langchain-community` shim, we implement the two-method
  interface (`embed_documents`, `embed_query`) ourselves. This is the exact
  shape any embedding provider must satisfy.
- **A confidence guardrail** (`MIN_SCORE`). The single most common failure of
  RAG demos is confidently answering from an irrelevant passage. We map Chroma's
  distance to a real cosine similarity and refuse below a threshold.
- **An eval harness** (`eval.py`). A fixed list of questions with known correct
  source files, scored automatically. This is the "eval" box on any serious
  AI-engineering checklist.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# build the index from data/
python -m vitos.ingest

# ask questions (retrieval-only, no API key)
python app.py

# score it
python eval.py
```

### Optional: generated answers (`--llm`)

```bash
pip install langchain-openai

# OpenAI
export OPENAI_API_KEY=sk-...
python app.py --llm

# fully local with Ollama
ollama pull llama3.1 && ollama serve
export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama LLM_MODEL=llama3.1
python app.py --llm
```

---

## The knowledge base

`data/` holds six markdown files — the kind of content a real shop would have:
`hours.md`, `menu.md`, `allergens.md`, `ordering.md`, `catering.md`,
`payment.md`. Swap in any business's docs and the assistant adapts with no code
changes.

---

## Experiments

1. **Chunk size.** In `vitos/config.py`, set `CHUNK_SIZE = 200` then re-ingest
   and re-run `eval.py`. Watch the accuracy move.
2. **Top-k.** Set `TOP_K = 2` vs `8` and re-run the eval.
3. **The guardrail.** Set `MIN_SCORE = 0.0` and ask the off-topic question —
   see it confidently answer nonsense. Then restore it.
4. **Add a question.** Append a row to `CASES` in `eval.py` and see whether the
   retriever finds the right source.

---

## What this demonstrates (for interviews)

- End-to-end RAG: load → chunk → embed → store → retrieve → generate
- LangChain + a vector database (Chroma), the stack most postings name
- Grounding + citation, not free-form hallucination
- A guardrail for out-of-scope questions
- An evaluation harness with a fixed test set and a scored metric
