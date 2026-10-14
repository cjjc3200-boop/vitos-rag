"""Central config — paths, model names, and retrieval knobs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / ".chroma"          # persistent vector store on disk
COLLECTION = "vitos"

# Local embedding model (no API key needed to build the index)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- Chunking knobs (RecursiveCharacterTextSplitter) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- Retrieval ---
TOP_K = 4

# Guardrail: minimum cosine similarity for a confident answer. Below this the
# assistant says "I don't know" instead of answering from an irrelevant passage.
# Empirically, on-topic questions score ~0.6+ and off-topic ones ~0.44.
MIN_SCORE = 0.50
