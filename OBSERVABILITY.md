# Observability — Frost Arena RAG

Every `/ask` request is traced in [Langfuse](https://langfuse.com): the prompt,
the response, latency, token counts, and cost — captured across the whole
multi-step flow, with each stage nested under a single parent span so you can
see where the time and money actually go.

## Setup

Provide these as environment variables (locally in a git-ignored `.env`; in
production as the host's secrets):

```bash
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Then open your Langfuse project; traces appear under the `ask` name.

## How it's instrumented

**Auto-tracing via a drop-in client.** When `LANGFUSE_PUBLIC_KEY` is set, the
app imports `OpenAI` from `langfuse.openai` instead of `openai`. It's a
subclass with the same constructor, so every `chat.completions.create` call is
traced automatically — generation, the injection classifier, and the grounding
checker all show up with their own model, tokens, and cost, no extra code at
each call site.

**One explicit parent span, everything nested under it.** The `/ask` handler
opens a single `ask` span and each stage runs inside its own child
observation, so a trace reads as a tree rather than a flat list of calls:

```
ask
├── retrieve                 (chunks + distances; retrieval latency in isolation)
├── guardrail:injection      (classifier verdict)
├── <generation>             (auto-nested by the traced client)
└── guardrail:grounding      (grounded / not, with reason)
```

Guardrail rejections annotate the parent span with the reason (`off_topic`,
`injection`, `ungrounded`, `generation_error`) so blocked requests are visible
in the dashboard *with why they were blocked*, not just absent.

## Fail-open by design

The app's LLM calls must never depend on the observability backend being
reachable. If `LANGFUSE_PUBLIC_KEY` is unset — in CI, or if Langfuse is down —
the app falls back to the plain `openai` client and keeps serving. Tracing is
an overlay, never a hard dependency on the request path.

## What the traces showed

Reading the dashboard turned two things from guesses into numbers:

- **Generation dominates latency (~80–90%).** Retrieval against the vector
  store is negligible by comparison — so any latency optimisation effort
  belongs on the generation/guardrail LLM calls, not on retrieval.
- **The full guardrail stack roughly tripled per-request cost**
  (~$0.00013 vs ~$0.00003 bare). That's the injection classifier plus the
  grounding checker (the latter on the larger 120b model) earning their cost.
  It's a deliberate, *measured* safety/cost trade-off rather than an assumption
  — and because the classifiers reuse models already priced in Langfuse, their
  cost is tracked for free.

## Reading a trace

Open any `ask` trace and you can see, top to bottom: the (PII-redacted)
question, each retrieved chunk and its distance, the injection verdict, the
generated answer with its token/cost figures, and the grounding verdict. A
cache hit short-circuits the whole tree and is tagged `cache: hit` on the
parent, so repeat questions are distinguishable from fresh ones at a glance.