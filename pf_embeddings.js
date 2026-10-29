// pf_embeddings.js
// Promptfoo embedding provider for the `similar` assertion.
// Runs in-process in Node (no Python worker), and calls your RAG app's /embed
// endpoint, so it reuses the all-MiniLM-L6-v2 model already loaded in your server.

module.exports = class LocalEmbeddingProvider {
  id() {
    return "local-minilm-embed";
  }

  // Required to exist; not used for similarity scoring.
  async callApi(prompt) {
    return { output: "" };
  }

  // Promptfoo calls this for `similar`. Return { embedding: number[] }.
  async callEmbeddingApi(text) {
    const url = `http://127.0.0.1:8000/embed?q=${encodeURIComponent(text)}`;
    const res = await fetch(url);
    if (!res.ok) {
      return { error: `embed endpoint returned ${res.status}` };
    }
    const data = await res.json();
    return { embedding: data.embedding };
  }
};
