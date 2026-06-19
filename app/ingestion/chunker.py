from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 120):
    """
    Splits raw text into overlapping chunks.

    chunk_size and chunk_overlap matter a lot for retrieval quality:
    - too small -> chunks lose context, answers feel fragmented
    - too large -> retrieval gets noisy, irrelevant text dilutes the match
    - overlap prevents a sentence from being cut exactly at a chunk boundary
      and losing meaning on both sides
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        {"content": chunk, "source": source, "chunk_id": f"{source}::{i}"}
        for i, chunk in enumerate(chunks)
    ]
