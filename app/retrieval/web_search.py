from tavily import TavilyClient
from langfuse import observe

from app.config import settings

_client = None


def get_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    return _client


@observe()
def web_search(query: str, max_results: int = 5):
    """search_depth='advanced' costs more credits but returns much better
    content extraction than 'basic' - worth it for a research agent."""
    client = get_client()
    response = client.search(query=query, max_results=max_results, search_depth="advanced")

    return [
        {
            "content": r["content"],
            "source": r["url"],
            "score": r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]
