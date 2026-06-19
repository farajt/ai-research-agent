from fastapi import FastAPI
from pydantic import BaseModel

from app.retrieval.web_search import web_search
from app.retrieval.vector_search import vector_search
from app.generation.synthesizer import synthesize_answer

app = FastAPI(title="AI Research Agent - Week 1 (naive baseline)")


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(request: QueryRequest):
    """
    Week 1 baseline: no query rewriting, no reranking, no guardrails yet.
    This intentionally simple version is what we measure week 2/3 improvements against.
    """
    web_results = web_search(request.question, max_results=5)
    vector_results = vector_search(request.question, top_k=5)
    all_chunks = web_results + vector_results

    result = synthesize_answer(request.question, all_chunks)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
