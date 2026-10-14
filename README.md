# Vito's Sandwich Shop — RAG Customer Assistant

A production-shaped Retrieval-Augmented Generation (RAG) assistant for a
restaurant, built with the applied-AI stack that job postings actually name:
**LangChain + Chroma (vector database) + RAG**, plus **automated evals**,
**tracing**, and a **deployable HTTP API**.

The "client" is a mock Tampa Bay sandwich shop. The point of the project is to
demonstrate the whole pipeline *and* the things that separate a hireable
candidate from a hobbyist: **grounding, citations, a confidence guardrail,
evaluation, observability, and deployment.**

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
| **Trace** | Time and record each stage per request | `vitos/tracing.py` |
| **Serve** | Expose it as an HTTP API | `api.py` |

### The deliberate design choices

- **A hand-written LangChain `Embeddings` class** (`embeddings.py`). Rather than
  depend on the sunset `langchain-community` shim, we implement the two-method
  interface (`embed_documents`, `embed_query`) ourselves. This is the exact
  shape any embedding provider must satisfy.
- **A confidence guardrail** (`MIN_SCORE`). The single most common failure of
  RAG demos is confidently answering from an irrelevant passage. Because our
  vectors are unit-length, Chroma's squared-L2 distance converts to a real
  cosine similarity (`cos = 1 − d²/2`); we refuse below a threshold.
- **An eval harness** (`eval.py`). A fixed list of questions with known correct
  source files, scored automatically — 10/10 on this knowledge base.
- **Structured tracing** (`tracing.py`). Every request emits a JSON trace with a
  timed span per stage, retrieved sources, the top score, and total latency.
- **A live API** (`api.py`). FastAPI + Docker, index built at image-build time.

---

## Quickstart (CLI)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m vitos.ingest             # build the index from data/
python app.py                      # ask questions (retrieval-only, no key)
python eval.py                     # score it
```

## Run it as an API

```bash
pip install -r requirements-api.txt
uvicorn api:app --reload --port 8000

curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"do you have gluten free bread?"}'
```

Response (abridged):

```json
{
  "question": "do you have gluten free bread?",
  "answered": true,
  "sources": ["allergens.md", "menu.md"],
  "top_score": 0.7121,
  "latency_ms": 19.42
}
```

### Run it with Docker

```bash
docker build -t vitos-api .
docker run -p 8000:8000 vitos-api
# GET localhost:8000/health  ->  {"status":"ok","index_loaded":true,...}
```

### Optional: generated answers (`--llm` / `VITOS_LLM=true`)

```bash
pip install langchain-openai

# OpenAI
export OPENAI_API_KEY=sk-...
python app.py --llm

# fully local with Ollama (no data leaves the machine)
ollama pull llama3.1 && ollama serve
export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama LLM_MODEL=llama3.1
python app.py --llm
```

---

## Observability

Two layers, both intentional:

1. **Built-in structured tracing** (`vitos/tracing.py`, always on). Each request
   logs a JSON line with a span per stage:

   ```json
   {"question":"do you have gluten free bread?","answered":true,"top_score":0.7121,
    "sources":["allergens.md","menu.md"],
    "spans":[{"name":"retrieve","duration_ms":12.1,"attributes":{"top_k":4}},
             {"name":"generate","duration_ms":7.3,"attributes":{"mode":"retrieval"}}]}
   ```

2. **Optional LangSmith tracing.** Set `LANGCHAIN_TRACING_V2=true` and
   `LANGCHAIN_API_KEY=...` and LangChain emits full traces automatically.
   `GET /health` reports whether it is on.

---

## The knowledge base

`data/` holds six markdown files — the kind of content a real shop would have:
`hours.md`, `menu.md`, `allergens.md`, `ordering.md`, `catering.md`,
`payment.md`. Swap in any business's docs and the assistant adapts with no code
changes.

---

## Experiments

1. **Chunk size.** In `vitos/config.py`, set `CHUNK_SIZE = 200`, re-ingest, and
   re-run `eval.py`. Watch the accuracy move.
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
- Structured tracing for latency and retrieval quality per request
- A containerized HTTP API — deployable, not just runnable
