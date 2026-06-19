import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.loader import load_text_files
from app.ingestion.chunker import chunk_text
from app.ingestion.embed_store import add_chunks
from app.ingestion.bm25_store import add_chunks_to_corpus

RAW_DIR = "data/raw"


def main():
    documents = load_text_files(RAW_DIR)
    if not documents:
        print(f"No .txt files found in {RAW_DIR}. Add some documents there first.")
        return

    total_chunks = 0
    for doc in documents:
        chunks = chunk_text(doc["text"], source=doc["source"])
        add_chunks(chunks)
        add_chunks_to_corpus(chunks)
        total_chunks += len(chunks)
        print(f"Ingested {doc['source']}: {len(chunks)} chunks")

    print(f"\nDone. Total chunks ingested: {total_chunks}")


if __name__ == "__main__":
    main()
