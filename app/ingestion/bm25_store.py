import json
import os

from rank_bm25 import BM25Okapi

from app.config import settings

CORPUS_PATH = os.path.join(settings.CHROMA_PERSIST_DIR, "bm25_corpus.json")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _load_corpus() -> list[dict]:
    if not os.path.exists(CORPUS_PATH):
        return []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def add_chunks_to_corpus(chunks: list[dict]):
    """Keeps a flat JSON corpus alongside the vector store, keyed by chunk_id
    so re-ingesting the same file overwrites rather than duplicates."""
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    existing_by_id = {c["chunk_id"]: c for c in _load_corpus()}
    for c in chunks:
        existing_by_id[c["chunk_id"]] = c

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(existing_by_id.values()), f)


def bm25_search(query: str, top_k: int = 8):
    """Rebuilds the BM25 index from the persisted corpus on every call.
    Fine at this corpus size (hundreds of chunks). At real scale this index
    should be built once and cached, rebuilt only when the corpus changes -
    a known, intentional scaling limitation for this stage, not an oversight.
    """
    corpus = _load_corpus()
    if not corpus:
        return []

    tokenized_corpus = [_tokenize(c["content"]) for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {"content": c["content"], "source": c["source"], "score": float(score)}
        for c, score in ranked
        if score > 0
    ]
