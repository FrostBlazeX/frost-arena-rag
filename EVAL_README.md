# Evaluation — Frost Arena RAG

A re-runnable Promptfoo suite that grades the live `/ask` endpoint against a
fixed set of questions, and a CI gate that blocks any change that breaks it.
The point is not "does the LLM sound plausible" but "does the whole pipeline —
retrieval, guardrails, generation — still return the right answer, provably,
on every push."

Config: [`promptfooconfig.yaml`](./promptfooconfig.yaml) · CI: [`.github/workflows/eval.yml`](./.github/workflows/eval.yml)

## Running it locally

The suite calls the API over HTTP, so the server has to be up first:

```bash
# 1. start the API (needs GROQ_API_KEY in your environment / .env)
uvicorn main:app --host 127.0.0.1 --port 8000

# 2. in another shell, run the suite
npx promptfoo@latest eval -c promptfooconfig.yaml --no-cache -j 1
npx promptfoo@latest view        # open the results UI
```

`-j 1` serialises the calls so the suite doesn't burst past Groq's rate limit.
`--no-cache` makes each run a true end-to-end test rather than a replay.

The provider points at `http://127.0.0.1:8000/ask` and reads `json.answer` out
of the response, so the suite grades exactly what a real client receives.

## Metric strategy: the right tool per assertion

The governing rule is that **an LLM judge only earns its keep when the output
is variable.** Grading a fixed string with a model just adds flakiness and
token cost, so each case uses the cheapest metric that actually tests the
contract.

| Metric                | Used for                                 | Why                                                                          |
| --------------------- | ---------------------------------------- | ---------------------------------------------------------------------------- |
| `regex` / `icontains` | Exact facts (times, prices)              | Deterministic; `5\s*pm` tolerates "5pm"/"5 pm" spacing                       |
| `similar` (semantic)  | Topical prose (e.g. the location answer) | Rewording shouldn't fail the test; threshold 0.6 via `pf_embeddings.js`      |
| `llm-rubric`          | Open-ended quality (quiz-night answer)   | Judges _correctness of claim_, not wording — the one genuinely variable case |
| `icontains-any`       | Guardrail refusals                       | The guardrail emits a **fixed** canned string, so match it directly          |

## The 8 cases (11 assertions)

**Fact retrieval** — happy-hour start/end (`5pm`/`7pm`), monthly membership
(`£25`), under-18s cutoff (`8pm`).

**Topical prose** — venue location, graded by semantic similarity so a
reworded-but-correct answer passes.

**Fact + quality** — quiz night: a deterministic `Wednesday` check _and_ an
`llm-rubric` that confirms the model doesn't invent a different day/time.

**Grounding / safety** — three adversarial prompts (an off-KB question, a
secret-seeking question, and a direct "ignore the context" injection) that must
hit a guardrail and return a canned refusal rather than invent or leak.

## Two decisions worth calling out

**The judge is a bigger model than the app.** The app generates with
`gpt-oss-20b`; the `llm-rubric` judge runs on `gpt-oss-120b`. A judge weaker
than the system it grades produces false negatives — the evaluator should be
more reliable than the thing under test.

**Guardrail refusals are graded as fixed strings, not by a rubric.** An earlier
version used `llm-rubric` for the refusals, but the rubric described a _soft_
refusal ("says it doesn't know") while the app actually emits a _scope_ refusal
("I can only answer questions about The Frost Arena"). That mismatch left even
the 120b judge on the fence, so identical output graded PASS one run and FAIL
the next. A blocking CI gate has to be stable, so the refusals are now matched
with `icontains-any` against the guardrail's real output. `icontains-any` also
keeps the case green if a prompt trips a _different_ guardrail than expected
(relevance vs. injection), since it asserts the contract — "a refusal was
returned" — not the specific path taken.

## CI gate

`.github/workflows/eval.yml` runs on every push and PR: it installs deps,
starts the API, waits for `/docs` to respond, runs the suite to `results.json`,
and **fails the build if `failures` or `errors` is non-zero.** A green check
means the full pipeline still answers every case correctly; a red one blocks
the merge.
