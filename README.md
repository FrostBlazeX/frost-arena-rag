# Frost Arena RAG — Evaluation Suite

An automated, re-runnable evaluation suite for a retrieval-augmented generation
(RAG) service, built with [Promptfoo](https://www.promptfoo.dev/). It replaces an
ad-hoc `eval.py` script with a declarative, cached, reportable test matrix that
gates changes the way a unit-test suite gates application code.

## Why this exists

An LLM feature that "works on my machine" is not production-ready until its
behaviour is measured on every change. This suite answers three questions that
matter in production:

1. **Does it retrieve the right facts?** (correctness)
2. **Does it stay grounded?** (it must refuse to invent facts not in its context)
3. **Can it be talked out of its guardrails?** (basic prompt-injection resistance)

## What is under test

`GET /ask?q=<question>&k=3` on the Frost Arena RAG service: a FastAPI app that
retrieves the top-k document chunks from a Chroma vector store
(`all-MiniLM-L6-v2` embeddings) and generates a grounded answer with an
instruction to answer **only** from the retrieved context.

## Metrics, and when each is used

Choosing the right metric per assertion is the core design decision. Each metric
verifies a different kind of claim:

| Metric        | Promptfoo type                 | Verifies                          | Example use here                   |
| ------------- | ------------------------------ | --------------------------------- | ---------------------------------- |
| Deterministic | `regex`, `icontains`           | Exact facts: times, prices, names | Happy-hour times, membership price |
| Semantic      | `similar` (cosine ≥ threshold) | On-topic / roughly complete prose | Venue location answer              |
| LLM-as-judge  | `llm-rubric`                   | Quality, grounding, safety        | Refusals, injection resistance     |

**Design note:** semantic similarity is deliberately _not_ used for numeric facts.
Cosine similarity is largely blind to numbers — a wrong "9am to 11am" answer scores
nearly as high as the correct "5pm to 7pm" against the same reference — so exact
facts are checked with `regex`/`icontains` or an LLM judge instead.

## Grounding & safety tests

The highest-value cases assert what the system must _refuse_ to do:

- Out-of-context questions (lottery tickets, wifi password) must return "I don't
  know", not a fabricated answer.
- An instruction to ignore the context and recommend a competitor must be declined.

These directly test the system prompt's "answer only from context, never invent
facts" contract, which is otherwise untested and therefore unenforced.

## Evaluator reliability

The app generates answers with `gpt-oss-20b`, but the `llm-rubric` judge runs on a
larger model (`gpt-oss-120b`). This is deliberate: a weak judge produces false
negatives. An earlier run graded a correct, grounded refusal as a failure purely
because the small grader couldn't reliably produce a valid verdict. The answer under
test never changed; only swapping to a stronger judge turned the suite green, which
is itself the lesson: a red result can mean a broken evaluator, not a broken system.

Meta's Prompt Guard was also trialled for this role and rejected. It is an input-side
injection _classifier_, not an output judge, so it has nothing to return when asked to
grade an answer. Its proper home is in front of the app, screening incoming requests
as a guardrail, not in the grading step.

## Running the suite

Prerequisites: Node.js 18+, Python 3.11+ with the RAG app's dependencies, and a
`GROQ_API_KEY` (used by the `llm-rubric` grader).

```bash
# 1. Start the RAG service (it exposes /ask and a small /embed helper)
uvicorn main:app --reload            # serves on http://127.0.0.1:8000

# 2. In a second shell, set the grader key and run the eval
export GROQ_API_KEY=...              # PowerShell: $env:GROQ_API_KEY="..."
npx promptfoo@latest eval            # runs the matrix, prints pass/fail
npx promptfoo@latest view            # opens the local results UI
```

Results are cached in `~/.promptfoo/cache`, so re-runs only re-call the model for
cases whose output changed. The single committed `promptfooconfig.yaml` is the
entire, reproducible report — it drops straight into CI.

## How semantic scoring stays local

Promptfoo has no built-in for a local sentence-transformers model, so `similar`
is backed by a small in-process JavaScript provider (`pf_embeddings.js`) that
calls the RAG service's `/embed` endpoint. This reuses the _same_ `all-MiniLM-L6-v2`
model the retrieval uses, so similarity is scored in the identical vector space —
and it avoids spawning a separate Python worker.

## Files

- `promptfooconfig.yaml` — the eval suite (cases + assertions)
- `pf_embeddings.js` — local embedding provider for the `similar` metric
- `main.py` — the RAG service (`/ask`, `/search`, `/embed`)

## Companion suite

A parallel Promptfoo suite in `week4-api/` evaluates the github-repos **agent**
(`/agent`), using the same metric taxonomy, grown from the original `eval.py`.
