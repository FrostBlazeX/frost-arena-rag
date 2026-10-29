# Frost Arena RAG

A production-grade retrieval-augmented generation service that answers questions
about a venue ("The Frost Arena") using only a trusted knowledge base. Built to
demonstrate the full lifecycle of an LLM feature: not just working, but evaluated,
observable, guarded, and deployed.

## Live demo

**Live:** https://frost-arena-rag.onrender.com

- Interactive API docs: https://frost-arena-rag.onrender.com/docs
- Ask a question: https://frost-arena-rag.onrender.com/ask?q=when is happy hour

> Hosted on Render's free tier, which sleeps when idle, so the first request after
> a pause takes ~30-60s to wake and load the embedding model. Subsequent requests
> are fast.

## Architecture

```
question ─▶ relevance gate ─▶ injection gate ─▶ retrieve (Chroma) ─▶ generate (Groq)
                │                   │                                      │
             (distance)        (classifier)                        grounding check
                │                   │                                      │
                └──── reject off-topic / unsafe ──┐                   PII redaction
                                                   ▼                       ▼
                                           safe response           answer + sources
```

- **Retrieval:** Chroma with `all-MiniLM-L6-v2` embeddings (ONNX build), rebuilt in
  memory on startup, no external database.
- **Generation:** Groq `openai/gpt-oss-20b` via the OpenAI-compatible API.
- **API:** FastAPI (`/ask`, `/search`, `/embed`).

## The three production pillars

1. **Evaluation** — a re-runnable Promptfoo suite with three metric types
   (deterministic, semantic similarity, LLM-as-judge) plus grounding and injection
   tests. See `EVAL_README.md`.
2. **Observability** — every request traced in Langfuse: prompt, response, latency,
   tokens, and cost, across the multi-step flow. See `OBSERVABILITY.md`.
3. **Guardrails** — input: a free distance-based relevance gate + an LLM injection
   classifier; output: graceful fallback, a grounding check, and PII redaction.

## Results

- Eval suite passes with each metric applied where it fits (e.g. regex for exact
  facts, semantic similarity for prose, LLM-judge for grounding and safety).
- Tracing showed generation dominates latency (~80-90%); retrieval is negligible.
- The full guardrail stack roughly tripled per-request cost (~$0.00013 vs ~$0.00003
  bare), a deliberate, measured safety/cost trade-off.
- Off-topic and injection attempts are rejected before reaching the model, visible
  in the dashboard with the reason each request was blocked.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Provide these as environment variables (locally in a git-ignored `.env`; in
production as the host's secrets):

```
GROQ_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Deployment

Runs as a Python web service on Render's free tier. Notes that made it fit:

- Uses Chroma's **ONNX** embedder (not the torch SentenceTransformer) to stay under
  the 512MB free-tier memory limit.
- Secrets are set as Render environment variables, never committed.
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- The relevance-gate threshold is calibrated to the ONNX embedder's distance scale
  (on-topic ~0.6, off-topic ~1.9; threshold 1.2).
