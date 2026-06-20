import json

from langchain_groq import ChatGroq
from langfuse import observe

from app.config import settings

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.LLM_MODEL, temperature=0.0)
    return _llm


SYSTEM_PROMPT = """You rewrite user questions into 1-3 focused search queries optimized
for retrieval. If the question is already simple and specific, return it unchanged as a
single query. If it bundles multiple questions or uses vague phrasing, decompose it into
clearer, more specific sub-queries.

Respond with ONLY a JSON array of strings, nothing else. No markdown, no explanation.
Example: ["query one", "query two"]"""


@observe()
def rewrite_query(question: str) -> list[str]:
    llm = get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    )

    raw = response.content.strip()
    # Models sometimes wrap JSON in markdown fences despite instructions not to -
    # strip defensively rather than letting a formatting quirk break the pipeline
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        queries = json.loads(raw)
        if isinstance(queries, list) and queries and all(isinstance(q, str) for q in queries):
            return queries
    except json.JSONDecodeError:
        pass

    # Fallback: if parsing fails for any reason, fall back to the original
    # question rather than letting the whole pipeline error out
    return [question]
