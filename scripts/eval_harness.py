import sys
import os
import time
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.web_search import web_search
from app.retrieval.vector_search import vector_search
from app.retrieval.hybrid_search import hybrid_local_search
from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.reranker import rerank
from app.generation.synthesizer import synthesize_answer
from app.guardrails.citation_checker import check_groundedness

# A small, fixed test set. Each question has a few keywords that a correct,
# on-topic answer should mention - a simple, defensible proxy for "did
# retrieval actually find the right information," without needing a full
# hand-labeled relevance-judged dataset (which is real IR-eval territory,
# out of scope for a project at this stage - this is explicitly a
# lightweight proxy, not a TREC-style benchmark).
EVAL_QUESTIONS = [
    {
        "question": "What is the difference between a bi-encoder and a cross-encoder?",
        "expected_keywords": ["bi-encoder", "cross-encoder", "separately", "together"],
    },
    {
        "question": "What is hybrid search in information retrieval?",
        "expected_keywords": ["bm25", "dense", "vector", "keyword"],
    },
    {
        "question": "Why is reranking used after initial retrieval?",
        "expected_keywords": ["rerank", "precision", "candidates"],
    },
    {
        "question": "What is Reciprocal Rank Fusion?",
        "expected_keywords": ["rank", "fusion", "merge"],
    },
    {
        "question": "What does it mean for an AI answer to be grounded in its sources?",
        "expected_keywords": ["citation", "source", "support"],
    },
]

# Free-tier LLM rate limits are easy to hit when running ~25 calls back to
# back (5 questions x 2 pipelines x ~2-3 LLM calls each). A small delay
# between runs spreads out token usage instead of bursting it.
SLEEP_BETWEEN_RUNS_SECONDS = 10


def _dedupe(chunks: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for c in chunks:
        key = f"{c['source']}::{c['content'][:80]}"
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def run_naive(question: str) -> dict:
    """Calls the internal pipeline functions directly rather than the HTTP
    endpoint, so we have access to the actual retrieved chunks - needed to
    run the guardrail check, which the naive /query/naive HTTP response
    doesn't expose."""
    start = time.perf_counter()
    web_results = web_search(question, max_results=5)
    vector_results = vector_search(question, top_k=5)
    chunks = web_results + vector_results

    result = synthesize_answer(question, chunks)
    guardrail = check_groundedness(result["answer"], chunks)
    latency = time.perf_counter() - start

    return {
        "answer": result["answer"],
        "num_sources": len(result["sources"]),
        "latency_seconds": round(latency, 2),
        "guardrail": guardrail,
    }


def run_hybrid(question: str) -> dict:
    start = time.perf_counter()
    sub_queries = rewrite_query(question)

    all_candidates = []
    for sq in sub_queries:
        all_candidates.extend(web_search(sq, max_results=5))
        all_candidates.extend(hybrid_local_search(sq, top_k=8))

    deduped = _dedupe(all_candidates)
    top_chunks = rerank(question, deduped, top_k=6)

    result = synthesize_answer(question, top_chunks)
    guardrail = check_groundedness(result["answer"], top_chunks)
    latency = time.perf_counter() - start

    return {
        "answer": result["answer"],
        "num_sources": len(result["sources"]),
        "latency_seconds": round(latency, 2),
        "guardrail": guardrail,
    }


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords actually present in the answer.
    A crude but honest proxy for 'did this answer actually address the
    question with relevant, specific information.'"""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0.0


def grounding_rate(guardrail: dict) -> float:
    """Fraction of checked claims that the guardrail confirmed were
    actually supported by their cited source."""
    checked = guardrail.get("checked_claims", 0)
    flagged = len(guardrail.get("flagged_claims", []))
    if checked == 0:
        return 1.0
    return (checked - flagged) / checked


def _avg(results: list[dict], key: str) -> float:
    values = [r[key] for r in results]
    return sum(values) / len(values) if values else 0.0


def main():
    results = {"naive": [], "hybrid": []}

    for item in EVAL_QUESTIONS:
        question = item["question"]
        keywords = item["expected_keywords"]
        print(f"\nRunning: {question}")

        print("  -> naive pipeline...")
        naive = run_naive(question)
        naive["keyword_coverage"] = keyword_coverage(naive["answer"], keywords)
        naive["grounding_rate"] = grounding_rate(naive["guardrail"])
        results["naive"].append({"question": question, **naive})
        time.sleep(SLEEP_BETWEEN_RUNS_SECONDS)

        print("  -> hybrid pipeline...")
        hybrid = run_hybrid(question)
        hybrid["keyword_coverage"] = keyword_coverage(hybrid["answer"], keywords)
        hybrid["grounding_rate"] = grounding_rate(hybrid["guardrail"])
        results["hybrid"].append({"question": question, **hybrid})
        time.sleep(SLEEP_BETWEEN_RUNS_SECONDS)

    print("\n" + "=" * 60)
    print("EVAL SUMMARY (naive vs hybrid)")
    print("=" * 60)
    print(f"{'Metric':<28}{'Naive':<15}{'Hybrid':<15}")
    print(
        f"{'Avg keyword coverage':<28}"
        f"{_avg(results['naive'], 'keyword_coverage'):<15.2f}"
        f"{_avg(results['hybrid'], 'keyword_coverage'):<15.2f}"
    )
    print(
        f"{'Avg grounding rate':<28}"
        f"{_avg(results['naive'], 'grounding_rate'):<15.2f}"
        f"{_avg(results['hybrid'], 'grounding_rate'):<15.2f}"
    )
    print(
        f"{'Avg latency (s)':<28}"
        f"{_avg(results['naive'], 'latency_seconds'):<15.2f}"
        f"{_avg(results['hybrid'], 'latency_seconds'):<15.2f}"
    )
    print(
        f"{'Avg num sources':<28}"
        f"{_avg(results['naive'], 'num_sources'):<15.2f}"
        f"{_avg(results['hybrid'], 'num_sources'):<15.2f}"
    )

    os.makedirs("data", exist_ok=True)
    with open("data/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to data/eval_results.json")


if __name__ == "__main__":
    main()
