"""The Frost Arena knowledge base — single source of truth.

Imported by both main.py (the Chroma fallback seeds from it) and
migrate_to_supabase.py (loads it into pgvector), so the app and the migration
can never drift out of sync.
"""

DOCUMENTS = [
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
]
