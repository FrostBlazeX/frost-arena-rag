# Retrieval: Hybrid Search + Reranking

How `frost-arena-rag` finds the right document, built and measured in two stages.
Dense vector search is excellent at _meaning_ but blurs _exact tokens_ — product
codes, booking refs, IDs — once a corpus is large enough that near-identical
records compete. The fix is two complementary layers, each validated against a
re-runnable eval before it was trusted:

1. **Hybrid retrieval (recall)** — dense + keyword, fused with RRF, casting a
   wide net so the right document lands _somewhere_ in a candidate pool.
2. **Reranking (precision)** — an LLM reorders that pool so the right document
   lands _at the top_.

The headline, measured not assumed: overall retrieval went from **MRR 0.68 → 0.85**
(hit-rate@5 **71% → 90%**), hard exact-ID cases from **0.61 → effectively perfect**,
and the final system hits its own recall ceiling — every document the retriever
can reach, the reranker gets into the top 5.

## The pipeline

### Stage 1 — Hybrid retrieval

Both retrievers run inside the one Postgres (Supabase) that stores the documents.

- **Dense** — `match_documents`: pgvector cosine search over `all-MiniLM-L6-v2`
  embeddings. Captures meaning; "cheap evening drinks" ≈ "happy hour".
- **Keyword** — `match_documents_text`: Postgres full-text search. A `tsvector`
  generated column over `content` (GIN-indexed), matched against an OR-of-lexemes
  query ranked by `ts_rank`. Captures exact tokens the embedding smooths away.
- **Fusion** — Reciprocal Rank Fusion: each doc scores `Σ 1/(60 + rank)` across
  the two ranked lists. RRF fuses on _rank_, so it needs no score normalisation
  between two incompatible scales (cosine distance vs `ts_rank`).

### Stage 2 — Reranking

Dense and keyword are both **bi-encoders**: query and document are embedded
_separately_, then compared, so the encoder never sees the pair together and
can't reason about how they actually relate. A **reranker** scores `(query, doc)`
as a pair — far more accurate, but one model call per document, so it only runs
on a small candidate set. Hence the pattern:

> **Retrieve many (cheap, recall) → rerank few (expensive, precision).**

Here the reranker is an **LLM** (Groq `gpt-oss-20b`), listwise: it's shown the
query plus the numbered candidate pool and returns the passages best-first in one
call. A local cross-encoder would have been the textbook choice but needs torch,
which blows the 512MB Render budget; the LLM reranker reuses the stack already in
place and is traced in Langfuse, so its cost is measured, not guessed.

## How it's measured

`retrieval_eval.py` hits the live `/search` endpoint with a fixed query set and
reports, per query group:

- **hit-rate@k** — is the correct doc in the top-k at all (recall).
- **MRR** — mean reciprocal rank of the correct doc (how high it ranks).

Queries split into **EASY** (clean, semantic venue questions — dense's home turf)
and **HARD** (pinned to an opaque ID like `BK-7F3Q`, `S-204`, sitting in a cluster
of ~90 near-identical records).

### The corpus is deliberately hostile

The original 16-doc KB couldn't demonstrate anything — every fact lived in its
own clean sentence, dense scored MRR 0.95, and hybrid had no headroom. _That was
the first finding: measure before adding machinery._ To build an honest test the
KB was scaled to **321 documents** (16 curated facts + 5 fixed anchor records +
300 generated operational records: bookings, staff, menu, members, rooms), each
tagged with an opaque ID among look-alike siblings — the realistic condition where
hybrid and reranking earn their keep.

## Results

| Stage                         | EASY MRR  | HARD MRR  | OVERALL MRR | OVERALL hit@5 |
| ----------------------------- | --------- | --------- | ----------- | ------------- |
| Baseline (dense only)         | 0.714     | 0.607     | 0.679       | 71%           |
| Hybrid (equal-weight RRF)     | 0.657     | 0.905     | 0.740       | 81%           |
| Hybrid (tuned pools 20/5)     | 0.679     | 0.905     | 0.754       | 81%           |
| **+ Reranking (gpt-oss-20b)** | **0.821** | **0.893** | **0.845**   | **90.5%**     |

Reranking's final hit-rate@5 (90.5%) equals the measured **recall@50** — it
surfaced every document the retriever could reach. The system is now bounded by
_retrieval recall_, not by ranking.

## Two findings worth keeping

### Hybrid is not a free lunch

Adding the keyword retriever lifted HARD enormously (MRR 0.61 → 0.91) but _cost_
EASY (0.714 → 0.657): asked about "membership," keyword faithfully surfaces all
~50 member records that share the lexeme, and RRF blends them in, shoving the one
correct pricing doc down. Hybrid trades a little semantic precision for a large
exact-match recall gain. Tuning the candidate pools asymmetrically (dense 20,
keyword 5 — keyword as a targeted rescue, not a broad retriever) recovered most
of that cost. Reranking then removed it, because the reranker — not RRF — owns
final ordering, which freed the pools to widen back out (30/10) purely for recall.

### The reranker model: measured, not assumed

The obvious guess is "use the bigger model." The eval said otherwise. Reranking
the pool with `gpt-oss-120b` vs `gpt-oss-20b`:

|                | 120b                           | 20b               |
| -------------- | ------------------------------ | ----------------- |
| EASY MRR       | 0.768                          | **0.821**         |
| HARD MRR       | **1.000**                      | 0.893             |
| OVERALL MRR    | 0.845                          | 0.845             |
| rerank latency | ~1.1–4.4s (bimodal, p95 ~4.4s) | ~0.8–1.2s (tight) |
| cost / query   | higher                         | ~$0.00012         |

**Identical overall quality, at 3–4× the speed and a tighter latency
distribution.** The models trade strengths: 20b ranks semantic queries slightly
better (it kept the £25 membership doc at rank 1); 120b wins exactly one HARD case
— "extension for staff member S-204", where a decoy staff record's _extension
number_ is literally 204 and only the larger model disambiguates the ID from the
coincidental number. For this workload the decision is **ship 20b**, keeping 120b
noted as the fallback if exact-ID precision ever outranks latency.

## Known limitations

Two EASY queries — "do you have a loyalty scheme?" and "what's the address?" —
are missed by the _final_ system, and reranking cannot fix them: they aren't in
the candidate pool even at depth 50 (recall@50 = 90.5%, not 100%). "Loyalty
scheme" is pulled toward the 50 member records that literally contain "loyalty
points"; "what's the address?" is too short and generic for the bi-encoder to
place among 321 docs. These are **retrieval**, not ranking, problems — the next
lever is query expansion / multi-query retrieval, not a better reranker.

Reranking also adds latency on the hot path (~1s/query with 20b). Production
mitigations, in order of cheapness: skip reranking when the top dense result is
already high-confidence; shrink the rerank pool (the deepest reachable doc was
rank 20, so a pool of ~25 suffices); cache.

## Reproduce

```bash
# API up (Supabase backend), then:
python -I retrieval_eval.py              # top-5 quality
RAG_EVAL_K=50 python -I retrieval_eval.py # recall ceiling at depth
```

The query set and metrics are in `retrieval_eval.py`; the full-text schema and
both match functions are in the migration SQL; the RRF fusion and the LLM
reranker are in `main.py` (`_rrf_fuse`, `rerank`). Reranker latency, tokens and
cost are visible per request in Langfuse under the `rerank` span.
