# Hybrid Retrieval — dense + keyword, fused with RRF

Week 1 of the production-retrieval work. Dense vector search is excellent at
_meaning_ but blurs _exact tokens_ — product codes, booking refs, IDs — once a
corpus is large enough that near-identical records compete. This adds a keyword
retriever beside the vector one and fuses the two, recovering exact-match recall
without giving up semantic recall.

The headline result, measured not assumed: on a corpus engineered to expose the
weakness, hybrid lifted hard-case retrieval from **MRR 0.61 → 0.91** (hit-rate@3
**57% → 100%**) and overall retrieval from **MRR 0.68 → 0.75** — with a small
semantic-precision cost that was measured, understood, and mostly tuned back out.

## The retrieval stack

Both retrievers run inside the one Postgres (Supabase) that already stores the
documents — no second system to operate.

- **Dense** — `match_documents`: pgvector cosine search over `all-MiniLM-L6-v2`
  embeddings. Captures meaning; "cheap evening drinks" ≈ "happy hour".
- **Keyword** — `match_documents_text`: Postgres full-text search. A `tsvector`
  generated column over `content` (GIN-indexed) matched against an OR-of-lexemes
  query, ranked by `ts_rank`. Captures exact tokens the embedding smooths away.
- **Fusion** — Reciprocal Rank Fusion in `main.py`: each doc scores
  `Σ 1/(60 + rank)` across the two ranked lists; sort descending, take top-k. A
  doc that ranks well in _either_ list surfaces; one that ranks well in _both_
  wins. RRF fuses on _rank_, so it needs no score normalisation between two
  incompatible scales (cosine distance vs `ts_rank`).

## How it's measured

`retrieval_eval.py` hits the live `/search` endpoint with a fixed query set and
reports, per query group:

- **hit-rate@k** — is the correct doc in the top-k at all (recall).
- **MRR** — mean reciprocal rank of the correct doc (how high it ranks).

Queries are split into two groups:

- **EASY** — clean, semantic questions about the venue. Dense's home turf.
- **HARD** — questions pinned to an _opaque ID_ (`BK-7F3Q`, `S-204`,
  `SKU-30214`, …) sitting in a cluster of ~90 near-identical records.

### The corpus is deliberately hostile

The original 16-doc knowledge base could not demonstrate anything: every fact
lived in its own clean sentence, so dense retrieval scored a near-perfect MRR
0.95 and hybrid had no headroom. That itself was the first finding — _measure
before adding machinery._ To create an honest test, the KB was scaled to **321
documents** (16 curated venue facts + 5 fixed anchor records + 300 generated
operational records: bookings, staff, menu items, members, rooms), each tagged
with an opaque ID among many look-alike siblings. That is the realistic
condition — an operational dataset, not a tidy FAQ — where hybrid earns its
keep.

## Results

Baseline is dense-only; hybrid is dense + keyword + RRF.

| Query group     | Baseline (dense) | Hybrid (RRF) | Hybrid (tuned) |
| --------------- | ---------------- | ------------ | -------------- |
| **HARD** — MRR  | 0.607            | 0.905        | **0.905**      |
| HARD — hit@3    | 57%              | 100%         | **100%**       |
| OVERALL — MRR   | 0.679            | 0.740        | **0.754**      |
| OVERALL — hit@5 | 71%              | 81%          | **81%**        |
| EASY — MRR      | 0.714            | 0.657        | **0.679**      |

> **Tuned run** uses asymmetric candidate pools (dense 20 / keyword 5), making
> keyword a targeted exact-match rescue rather than a broad retriever. It held
> the HARD win exactly (MRR 0.905) while recovering most of the EASY regression
> (0.657 → 0.679): "How much is membership?" climbed from rank 5 back to rank 2.

Every opaque ID that dense missed — `S-204` and `R-14C` were retrieved at
**rank 1** by the keyword side; dense had not returned them in the top 5 at all.

## The tradeoff (the part worth reading)

Hybrid is **not** a free lunch, and the numbers say so: OVERALL and HARD rose,
but EASY _fell_ (0.714 → 0.657). The mechanism is clear from a single query.

"How much is membership?" ranks the correct pricing doc **#1 on dense**, but
**#5 on hybrid**. Asked about "membership," the keyword retriever faithfully
surfaces all ~50 `Member M-xxxxx holds an FA-GOLD membership` records — they
share the lexeme — and RRF blends those ranks in, shoving the one correct doc
down. On a query where keyword adds only noise, fusion costs precision.

So the honest characterisation: **hybrid trades a little semantic precision for
a large exact-match recall gain.** Net-positive here (OVERALL MRR +0.06, HARD
+0.30), but not strictly better everywhere — and the place it regresses is
exactly the place a keyword signal is least useful.

Tuning recovers most of that cost. Making keyword a narrow rescue (5 candidates,
not 20) stopped the bulk of the dilution: "How much is membership?" went from
rank 5 back to rank 2, lifting EASY from 0.657 to 0.679 and OVERALL to 0.754,
while HARD held at 0.905 exactly. The residual cost is now a single query one
rank below dense — a precision tax small enough to accept for a +0.30 HARD gain,
and one that reranking removes entirely.

The four EASY queries that neither retriever handles ("cheap drinks deals,"
"loyalty scheme," "what's the address," "somewhere to watch the football") share
no lexemes with their target docs _and_ get crowded in embedding space at scale.
Neither dense nor keyword can reach them — they are a _reranking_ problem, not a
retrieval one.

## What's next

- **Tuning (done):** asymmetric candidate pools (dense 20 / keyword 5) recovered
  most of the EASY regression (0.657 → 0.679) at zero cost to HARD (0.905).
- **Reranking (Week 2):** a cross-encoder reorders the fused candidate set by
  true query–document relevance. That recovers the semantic precision RRF costs
  _and_ is the only thing that can rescue the four lexeme-less EASY misses.

## Reproduce

```bash
# API up (Supabase backend), then:
python -I retrieval_eval.py
```

Baseline numbers were captured by pointing `/search` at dense-only
`match_documents`; hybrid by the RRF path in `main.py`. The query set and
metrics live in `retrieval_eval.py`; the full-text schema and both match
functions are in the migration SQL.
