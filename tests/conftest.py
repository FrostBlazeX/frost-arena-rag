
from chromadb.utils import embedding_functions
import chromadb
import os
from unittest.mock import patch


# The application requires this environment variable at import time.
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)


class FakeSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, **kwargs):
        pass


class FakeLangfuse:
    def start_as_current_observation(self, **kwargs):
        return FakeSpan()


class FakeCollection:
    def count(self):
        # Prevent main.py from inserting the knowledge base.
        return 1


class FakeChroma:
    def get_or_create_collection(self, *args, **kwargs):
        return FakeCollection()


with (
    patch("langfuse.get_client", return_value=FakeLangfuse()),
    patch.object(chromadb, "Client", return_value=FakeChroma()),
    patch.object(
        embedding_functions,
        "DefaultEmbeddingFunction",
        return_value=object(),
    ),
):
    import main
