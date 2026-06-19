from app.ingestion.embed_store import query_vector_store


def vector_search(query: str, top_k: int = 8):
    return query_vector_store(query, top_k=top_k)
