import json
import os
import sys
import urllib.parse
import urllib.request

BASE_URL = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
K = int(os.environ.get("RAG_EVAL_K", "5"))

# Each case: query, a substring that uniquely identifies the correct doc in
# kb.py (verified unique), and a group. Keep expect strings pinned to real,
# unique snippets — that's what makes the metric honest.
CASES = [
    # ---- EASY: clean, semantic. Dense should be near-perfect. ----
    {"q": "What time does happy hour start?",
        "expect": "happy hour",     "group": "easy"},
    {"q": "any cheap drinks deals in the evening?",
        "expect": "happy hour",     "group": "easy"},
    {"q": "How much is membership?",
        "expect": "£25 per month",  "group": "easy"},
    {"q": "do you have a loyalty scheme?",
        "expect": "£25 per month",  "group": "easy"},
    {"q": "Until what time are under-18s allowed?",
        "expect": "under-18s",      "group": "easy"},
    {"q": "can I bring my kids in the evening?",
        "expect": "under-18s",     "group": "easy"},
    {"q": "Where is the Frost Arena?",
        "expect": "Courtland Road", "group": "easy"},
    {"q": "what's the address?",
        "expect": "Courtland Road", "group": "easy"},
    {"q": "When is quiz night?",
        "expect": "7:30pm",         "group": "easy"},
    {"q": "is there a trivia evening?",
        "expect": "7:30pm",         "group": "easy"},
    {"q": "What time does happy hour end?",
        "expect": "7pm",            "group": "easy"},
    {"q": "somewhere to watch the football?",
        "expect": "big screens",    "group": "easy"},
    {"q": "where can I leave my car at night?",
        "expect": "40 spaces",      "group": "easy"},
    {"q": "when does the kitchen close?",
        "expect": "until 10pm",     "group": "easy"},

    # ---- HARD: opaque-id needles in large near-identical clusters. ----
    # Each id (BK-7F3Q, S-204, ...) sits among ~90 look-alike records. The id
    # barely moves the embedding, so dense drowns in near-neighbours; keyword
    # search nails the exact token. This is the canonical hybrid-wins case.
    {"q": "what does booking BK-7F3Q include?",
        "expect": "BK-7F3Q",   "group": "hard"},
    {"q": "what event is reservation BK-7F3Q for?",
        "expect": "BK-7F3Q",   "group": "hard"},
    {"q": "who is staff member S-204?",
        "expect": "S-204",     "group": "hard"},
    {"q": "what's the extension for staff member S-204?",
        "expect": "S-204",     "group": "hard"},
    {"q": "what is menu item SKU-30214?",
        "expect": "SKU-30214", "group": "hard"},
    {"q": "which membership tier does member M-10432 hold?",
        "expect": "M-10432",  "group": "hard"},
    {"q": "how many guests can room R-14C hold?",
        "expect": "R-14C",     "group": "hard"},
]


def search(query, k):
    url = BASE_URL + "/search?" + urllib.parse.urlencode({"q": query, "k": k})
    with urllib.request.urlopen(url, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def first_hit_rank(results, expect):
    needle = expect.lower()
    for i, row in enumerate(results, start=1):
        if needle in (row.get("document", "") or "").lower():
            return i
    return None


def summarise(rows, label):
    n = len(rows)
    if not n:
        return
    ks = sorted({k for k in (1, 3, 5, K) if k <= K})
    print(f"\n{label}  (n={n})")
    for k in ks:
        hits = sum(1 for r in rows if r and r <= k)
        print(f"  hit_rate@{k}: {hits / n:>7.2%}  ({hits}/{n})")
    mrr = sum((1.0 / r) if r else 0.0 for r in rows) / n
    print(f"  MRR       : {mrr:.3f}")


def main():
    print(f"Retrieval eval → {BASE_URL}  (k={K})\n")
    try:
        search("warmup", 1)
    except Exception as e:
        print(
            f"Could not reach {BASE_URL}/search — is the API running?\n  {e}")
        sys.exit(1)

    by_group = {}
    print(f"{'rank':>4}  {'grp':<4}  {'query':<42}  expect")
    print("-" * 78)
    for case in CASES:
        results = search(case["q"], K)
        rank = first_hit_rank(results, case["expect"])
        by_group.setdefault(case["group"], []).append(rank)
        shown = "miss" if rank is None else str(rank)
        q = case["q"] if len(case["q"]) <= 42 else case["q"][:39] + "..."
        print(f"{shown:>4}  {case['group']:<4}  {q:<42}  {case['expect']}")
    print("-" * 78)

    all_rows = [r for rows in by_group.values() for r in rows]
    summarise(by_group.get("easy", []), "EASY")
    summarise(by_group.get("hard", []),
              "HARD   <-- the number hybrid has to move")
    summarise(all_rows, "OVERALL")
    print("\nSave these. Re-run after hybrid + RRF and compare — watch the HARD group.")


if __name__ == "__main__":
    main()
