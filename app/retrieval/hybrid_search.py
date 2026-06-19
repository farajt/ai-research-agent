from app.retrieval.vector_search import vector_search
from app.ingestion.bm25_store import bm25_search


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merges multiple ranked lists into one using Reciprocal Rank Fusion:
    score = sum(1 / (k + rank)) across every list a document appears in.

    Why rank, not raw score: BM25 scores and cosine similarity scores live on
    completely different scales (BM25 can be 0-20+, cosine similarity is 0-1),
    so averaging them directly would let one method dominate arbitrarily.
    Using rank position instead of raw score sidesteps that entirely - a
    document that's #1 in both lists wins regardless of what the underlying
    scores actually were.
    """
    fused_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = f"{doc['source']}::{doc['content'][:50]}"
            doc_lookup[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    ranked_keys = sorted(fused_scores, key=lambda dk: fused_scores[dk], reverse=True)
    return [doc_lookup[dk] for dk in ranked_keys]


def hybrid_local_search(query: str, top_k: int = 10) -> list[dict]:
    """The 'true' hybrid search: dense (vector) + sparse (BM25) over the local
    knowledge base, merged via rank fusion. Distinct from merging web + vector
    as two separate sources, which main.py does separately at a higher level.
    """
    dense_results = vector_search(query, top_k=top_k)
    sparse_results = bm25_search(query, top_k=top_k)
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    return fused[:top_k]
