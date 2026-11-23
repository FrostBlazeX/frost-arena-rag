"""The Frost Arena knowledge base — single source of truth.

Imported by both main.py (the Chroma fallback seeds from it) and
migrate_to_supabase.py (loads it into pgvector), so the app and the migration
can never drift out of sync.

Two layers:
  CURATED       — the hand-written venue facts (clean, semantic, one fact each).
  OPERATIONAL   — a generated load of realistic records (bookings, staff, menu,
                  members, rooms), each keyed by an OPAQUE id in a crowd of
                  near-identical siblings. Deterministic (fixed seed), so every
                  import and every migration produces the identical corpus.

The operational layer exists to create honest retrieval difficulty: dense
embeddings blur one opaque id among ~90 look-alike records, which is where
hybrid (dense + keyword + RRF) starts to win. A handful of ANCHOR records have
FIXED ids so the eval suite can target them by id.
"""

import random

CURATED = [
    "The Frost Arena is a sports bar located at 26 Courtland Road, Oxford.",
    "The Frost Arena is open Mon-Thu 4pm-midnight, Fri-Sat noon-2am, and Sun noon-11pm.",
    "Happy hour runs every weekday from 5pm to 7pm, with 2-for-1 on draught beer and house cocktails.",
    "The Frost Arena has 18 big screens and two 4-metre projector walls, showing Premier League, NBA, UFC, and Formula 1.",
    "VIP booths seat up to 8 people and cost £120 to reserve on match days, including a £60 food and drink credit.",
    "Membership is £25 per month and includes priority booking, 10% off food, and a free drink on your birthday.",
    "The kitchen serves wings, loaded fries, burgers, and vegan options until 10pm every day.",
    "Under-18s are welcome until 8pm when accompanied by an adult.",
    "There is a weekly quiz night every Wednesday at 7:30pm, with a £50 bar-tab prize.",
    "Parking is free after 6pm in the rear car park, which has 40 spaces.",
    "FA-BRONZE membership costs £25 a month and includes priority booking and a birthday drink.",
    "FA-SILVER membership costs £45 a month and adds 10% off all food and drink.",
    "FA-GOLD membership costs £75 a month and includes unlimited VIP booth bookings plus a monthly guest pass.",
    "The signature cocktail list features the Frostbite Martini, the Blue Line Mule, and the Powerplay Punch.",
    "Private event bookings go through the events manager, Priya Chauhan, at events@frostarena.co.uk or 01865 555 0192.",
    "Screen 12, the main four-metre projector wall, can be hired for private viewing parties on non-match days.",
]

# --- Anchor records: FIXED opaque ids, referenced by retrieval_eval.py -------
ANCHORS = [
    "Booking BK-7F3Q is a private viewing-party reservation for the North London derby in VIP Booth 3 on 14 December, seating 8 guests, with a £60 bar credit.",
    "Staff member S-204 is Priya Chauhan, the events manager, reachable on extension 192.",
    "Menu item SKU-30214 is the Powerplay Punch, a £9 signature cocktail.",
    "Member M-10432 holds an FA-GOLD membership, active since March 2026, with 1,240 loyalty points.",
    "Room code R-14C is the upstairs mezzanine, hireable for private parties of up to 30 guests.",
]

# --- Generated distractors: same shape, different opaque ids ------------------
_EVENTS = [
    "the North London derby", "the Champions League final", "an NBA regular-season game",
    "a UFC pay-per-view", "the Formula 1 Grand Prix", "the Six Nations decider",
    "the Merseyside derby", "the Ryder Cup singles", "a heavyweight title fight",
    "the Cricket World Cup final", "a Premier League Monday-night game", "the Europa League final",
]
_BOOTHS = ["VIP Booth 1", "VIP Booth 2", "VIP Booth 3",
           "VIP Booth 4", "the mezzanine", "the terrace", "Screen 12"]
_FIRSTS = ["James", "Aisha", "Tom", "Grace", "Mohammed", "Chloe", "Daniel", "Sofia", "Liam", "Nadia",
           "Ben", "Yusuf", "Ella", "Marcus", "Leah", "Owen", "Zara", "Caleb", "Maya", "Hannah"]
_LASTS = ["Okafor", "Bennett", "Fletcher", "Hughes", "Abara", "Nowak", "Reilly", "Santos", "Khan",
          "Walsh", "Mensah", "Patel", "Doyle", "Osei", "Lambert", "Ryan", "Idris", "Barnes", "Flynn", "Hale"]
_ROLES = ["bar supervisor", "floor manager", "head chef", "sous chef", "duty manager",
          "security lead", "AV technician", "reservations host", "marketing coordinator", "assistant manager"]
_COCKTAILS = ["the Frostbite Martini", "the Blue Line Mule", "the Slapshot Spritz", "the Blue Line Negroni",
              "the Overtime Old Fashioned", "the Zamboni Zombie", "the Faceoff Fizz", "the Penalty Box Punch",
              "the Hat-Trick Highball", "the Power Play Spritz"]
_CATEGORIES = ["signature cocktail", "draught beer",
               "house wine", "alcohol-free cocktail", "sharing pitcher"]
_AREAS = ["the upstairs mezzanine", "the terrace bar", "the main sports floor", "the private screening room",
          "the ground-floor snug", "the rooftop deck"]
_TIERS = ["FA-BRONZE", "FA-SILVER", "FA-GOLD"]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _unique_codes(rng, prefix, body_fn, count, reserved):
    """Generate `count` unique '<prefix><body>' codes, avoiding `reserved`."""
    seen, out = set(reserved), []
    while len(out) < count:
        code = prefix + body_fn(rng)
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def _generate_operational(seed=42):
    rng = random.Random(seed)
    docs = []

    # Bookings (~90): the densest look-alike cluster — one opaque ref each.
    codes = _unique_codes(rng, "BK-",
                          lambda r: "".join(
                              r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4)),
                          90, {"7F3Q"})
    for c in codes:
        docs.append(
            f"Booking BK-{c[3:]} is a private viewing-party reservation for {rng.choice(_EVENTS)} "
            f"in {rng.choice(_BOOTHS)} on {rng.randint(1, 28)} {rng.choice(_MONTHS)}, "
            f"seating {rng.choice([4, 6, 8, 10, 12])} guests, with a £{rng.choice([40, 50, 60, 75, 100])} bar credit.")

    # Staff (~60)
    codes = _unique_codes(
        rng, "S-", lambda r: str(r.randint(100, 999)), 60, {"204"})
    for c in codes:
        name = f"{rng.choice(_FIRSTS)} {rng.choice(_LASTS)}"
        docs.append(f"Staff member S-{c[2:]} is {name}, {rng.choice(_ROLES)}, "
                    f"reachable on extension {rng.randint(100, 499)}.")

    # Menu items (~60)
    codes = _unique_codes(
        rng, "SKU-", lambda r: str(r.randint(10000, 99999)), 60, {"30214"})
    for c in codes:
        docs.append(f"Menu item SKU-{c[4:]} is {rng.choice(_COCKTAILS)}, "
                    f"a £{rng.choice([7, 8, 9, 10, 12])} {rng.choice(_CATEGORIES)}.")

    # Members (~50)
    codes = _unique_codes(
        rng, "M-", lambda r: str(r.randint(10000, 99999)), 50, {"10432"})
    for c in codes:
        docs.append(f"Member M-{c[2:]} holds an {rng.choice(_TIERS)} membership, "
                    f"active since {rng.choice(_MONTHS)} {rng.choice([2024, 2025, 2026])}, "
                    f"with {rng.randint(50, 2000)} loyalty points.")

    # Rooms (~40)
    codes = _unique_codes(rng, "R-",
                          lambda r: f"{r.randint(10, 39)}{r.choice('ABD')}", 40, {"14C"})
    for c in codes:
        docs.append(f"Room code R-{c[2:]} is {rng.choice(_AREAS)}, "
                    f"hireable for private parties of up to {rng.choice([15, 20, 25, 30, 40, 50])} guests.")

    return docs


DOCUMENTS = CURATED + ANCHORS + _generate_operational()

if __name__ == "__main__":
    # Quick sanity check: python kb.py
    print(f"{len(DOCUMENTS)} documents "
          f"({len(CURATED)} curated + {len(ANCHORS)} anchors + {len(DOCUMENTS) - len(CURATED) - len(ANCHORS)} generated)")
