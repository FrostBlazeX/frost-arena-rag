"""One-time migration: embed the Frost Arena knowledge base and load it into the
Supabase pgvector `documents` table.

Idempotent — clears the table first, so re-running never duplicates rows. Run it
once after provisioning, and again any time kb.py changes. Reads SUPABASE_URL and
SUPABASE_SECRET_KEY from the environment (.env).

    python migrate_to_supabase.py
"""
import os

from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from supabase import create_client

from kb import DOCUMENTS

load_dotenv()

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_SECRET_KEY"]      # secret key — server-side only
sb = create_client(url, key)

# Same embedder the app uses, so stored vectors match query vectors exactly.
embed_fn = embedding_functions.DefaultEmbeddingFunction()

# Clear existing rows so a re-run replaces rather than duplicates. (Supabase
# requires a filter on delete; id >= 0 matches every row.)
sb.table("documents").delete().gte("id", 0).execute()

vectors = embed_fn(DOCUMENTS)
rows = [{"content": doc, "embedding": [float(x) for x in vec]}
        for doc, vec in zip(DOCUMENTS, vectors)]
sb.table("documents").insert(rows).execute()

print(f"Loaded {len(rows)} documents into Supabase pgvector.")
