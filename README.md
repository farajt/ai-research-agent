# AI Research Agent — Week 1 (naive baseline)

A grounded AI research/search engine: hybrid retrieval (web + vector DB), reranking,
citation-level grounding, and hallucination guardrails. This is the week 1 milestone —
a working naive baseline (no reranking/guardrails yet) that later weeks will improve on
and measure against.

## Architecture (week 1 scope)

```
question -> web search (Tavily) ----\
                                      +--> LLM synthesis with citations -> answer
question -> vector search (Chroma) --/
```

Weeks 2-4 add: query rewriting, reranking, citation/hallucination guardrails,
observability tracing, an eval harness, and the frontend.

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your API keys:
   ```
   cp .env.example .env
   ```
   - Groq API key (free): https://console.groq.com
   - Tavily API key (free tier): https://tavily.com

3. Ingest the sample document into the vector store:
   ```
   python scripts/ingest.py
   ```
   (Drop your own `.txt` files into `data/raw/` first if you want different content.)

4. Run the API:
   ```
   uvicorn app.main:app --reload
   ```

5. Test it:
   ```
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the difference between a bi-encoder and a cross-encoder?"}'
   ```

   Or open http://localhost:8000/docs for the interactive Swagger UI.

## What's next (week 2)
- Query rewriting / decomposition before retrieval
- True hybrid search (BM25 + dense) inside the vector store, not just web + vector as two sources
- Cross-encoder reranking on merged candidates
