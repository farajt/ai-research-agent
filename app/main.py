from fastapi import FastAPI
from pydantic import BaseModel
from langfuse import observe

from app.retrieval.web_search import web_search
from app.retrieval.vector_search import vector_search
from app.retrieval.hybrid_search import hybrid_local_search
from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.reranker import rerank
from app.generation.synthesizer import synthesize_answer
from app.guardrails.citation_checker import check_groundedness

app = FastAPI(title="AI Research Agent - Week 2 (hybrid retrieval + reranking)")


class QueryRequest(BaseModel):
    question: str


def _dedupe(chunks: list[dict]) -> list[dict]:
    """Decomposing into sub-queries means the same chunk can come back
    multiple times across different sub-queries. Dedupe before reranking
    so the reranker isn't scoring the same content twice."""
    seen = set()
    deduped = []
    for c in chunks:
        key = f"{c['source']}::{c['content'][:80]}"
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


@app.post("/query")
@observe()
def query(request: QueryRequest):
    """
    Full week 2 pipeline:
    rewrite/decompose -> [web search + hybrid local search] per sub-query
    -> dedupe -> cross-encoder rerank -> grounded synthesis.
    """
    sub_queries = rewrite_query(request.question)

    all_candidates = []
    for sq in sub_queries:
        all_candidates.extend(web_search(sq, max_results=5))
        all_candidates.extend(hybrid_local_search(sq, top_k=8))

    deduped = _dedupe(all_candidates)
    top_chunks = rerank(request.question, deduped, top_k=6)

    result = synthesize_answer(request.question, top_chunks)
    result["sub_queries"] = sub_queries
    result["candidates_before_rerank"] = len(deduped)

    guardrail_result = check_groundedness(result["answer"], top_chunks)
    result["guardrail"] = guardrail_result

    return result


@app.post("/query/naive")
def query_naive(request: QueryRequest):
    """
    Week 1 baseline, kept as-is and reachable separately so the eval harness
    in week 3 can run both pipelines against the same question set and
    produce a real before/after comparison - not just a claim.
    """
    web_results = web_search(request.question, max_results=5)
    vector_results = vector_search(request.question, top_k=5)
    all_chunks = web_results + vector_results

    result = synthesize_answer(request.question, all_chunks)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
