# Vito's Sandwich Shop — RAG API
# Builds the index at image-build time so the container starts ready to serve.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first (layer caching: source changes won't re-run pip)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi 'uvicorn[standard]'

# Copy the project
COPY vitos/ ./vitos/
COPY data/ ./data/
COPY api.py eval.py README.md ./

# Build the vector index into the image
RUN python -m vitos.ingest

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
