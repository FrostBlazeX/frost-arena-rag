# Frost Arena RAG

A production-minded retrieval-augmented generation service that answers questions
about a venue ("The Frost Arena") using only a trusted knowledge base. Built as an
AI-engineering portfolio project to demonstrate the full lifecycle: not just a working
LLM feature, but one that is evaluated, observable, guarded, and deployed.

## Architecture

```
question ──▶ relevance gate ──▶ injection gate ──▶ retrieve (Chroma) ──▶ generate (Groq)
                 │                    │                                        │
              (distance)         (classifier)                          grounding check
                 │                    │                                        │
                 └──── reject off-topic / unsafe ────┐                    PII redaction
                                                      ▼                        ▼
                                              safe response            answer + sources
```

- **Retrieval:** Chroma vector store with `all-MiniLM-L6-v2` embeddings (rebuilt in
  memory on startup, no external database).
- **Generation:** Groq `openai/gpt-oss-20b` via the OpenAI-compatible API.
- **API:** FastAPI. Key endpoints: `GET /ask` (full RAG answer), `GET /search`
  (retrieval only), `GET /embed` (embedding helper).

## The three production pillars

1. **Evaluation** — a re-runnable Promptfoo suite (12 cases, three metric types:
   deterministic, semantic similarity, and LLM-as-judge) with grounding and safety
   tests. See `EVAL_README.md`.
2. **Observability** — every request is traced with Langfuse: prompt, response,
   latency, tokens, and cost, across the multi-step flow. See `OBSERVABILITY.md`.
3. **Guardrails** — input: a free distance-based relevance gate plus an LLM injection
   classifier; output: graceful fallback on model failure, a grounding check against
   the retrieved context, and PII redaction.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Set these as environment variables (locally in a git-ignored `.env`; in the Space,
under Settings → Variables and secrets):

```
GROQ_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Deployment

Runs as a Docker Space on Hugging Face (free CPU tier). The `Dockerfile` installs
dependencies and serves the app on port 7860; secrets are injected as environment
variables and never committed.
