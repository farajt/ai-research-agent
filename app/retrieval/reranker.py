from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 6) -> list[dict]:
    """Scores each (query, candidate) pair together in a single forward pass -
    far more precise than the bi-encoder/BM25 first-pass retrieval, but too
    slow to run across an entire corpus. That's why it only runs on the
    shortlist of candidates that already survived first-stage retrieval,
    rather than the whole database - the entire reason a two-stage pipeline
    exists at all.
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    reranked = []
    for c, score in scored[:top_k]:
        c_copy = dict(c)
        c_copy["rerank_score"] = float(score)
        reranked.append(c_copy)
    return reranked
