import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

# Lazy singletons: the embedding model is ~80MB and loading it on every
# request would be wasteful. Load once, reuse across requests.
_embedder = None
_client = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    return _embedder


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(name=settings.COLLECTION_NAME)
    return _collection


def add_chunks(chunks: list[dict]):
    """Embeds and stores a batch of chunks. Upsert means re-running ingestion
    on the same files won't create duplicates."""
    embedder = get_embedder()
    collection = get_collection()

    texts = [c["content"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    vectors = embedder.embed_documents(texts)
    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)


def query_vector_store(query: str, top_k: int = 8):
    """Embeds the query and returns the top_k nearest chunks by cosine distance."""
    embedder = get_embedder()
    collection = get_collection()

    query_vector = embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        output.append({"content": doc, "source": meta["source"], "score": 1 - dist})
    return output
