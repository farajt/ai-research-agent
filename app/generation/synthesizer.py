from langchain_groq import ChatGroq

from app.config import settings

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.LLM_MODEL, temperature=0.1)
    return _llm


def build_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[S{i}] (source: {c['source']})\n{c['content']}")
    return "\n\n".join(lines)


SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using ONLY the
provided source chunks below. Every factual claim must be followed by a citation marker
like [S1] or [S2] referencing the exact source chunk that supports it. If the sources do
not contain enough information to answer confidently, say so explicitly instead of guessing.
Do not invent citation numbers that were not provided to you."""


def synthesize_answer(question: str, chunks: list[dict]) -> dict:
    llm = get_llm()
    context = build_context(chunks)
    user_prompt = f"Question: {question}\n\nSources:\n{context}\n\nAnswer with citations:"

    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    return {
        "answer": response.content,
        "sources": [
            {"id": f"S{i}", "source": c["source"]} for i, c in enumerate(chunks, start=1)
        ],
    }
