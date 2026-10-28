# Resume Claim Evidence

This file separates the original reference implementation from the upgraded
target. A claim is resume-safe only after its acceptance test and evaluation are
reproducible.

## Multimodal agent

- Evidence: FastAPI chat and voice routes, `/voice/ws/{session_id}`, provider
  adapters for Whisper and Coqui XTTS, and WebSocket protocol tests.
- Measure: speech-to-text, first-response, synthesis, and end-to-end P50/P95.
- Do not claim a specific model was used unless the deployment configuration and
  benchmark output record that model.

## Long-term conversational memory

- Evidence: Redis session history, session summaries, TTL configuration, and
  session isolation tests.
- Measure: context-question accuracy and prompt character/token reduction against
  the buffer-only baseline.
- The original repository only had Redis buffer history and no summarization.

## Production-style RAG

- Evidence: semantic/recursive chunking, full Qdrant payloads, metadata filters,
  dense and lexical retrieval, reciprocal-rank fusion, and retrieval tests.
- Measure: Recall@k, MRR, grounded answer rate, and hallucination rate on a fixed
  labeled dataset.
- Changing embedding models requires matching `EMBEDDING_DIMENSION`.

## Retrieval optimization

- Evidence: content/URL deduplication, cosine scoring, source-quality and recency
  signals, top-k fusion, and optional reranker hooks in the live agent pipeline.
- Measure: compare each stage to a dense-only baseline using the same dataset.

## Live web grounding

- Evidence: query routing, SerpAPI normalization, source-quality/relevance
  ranking, citations, and failure fallback.
- Measure: factuality on time-sensitive questions with source and timestamp
  annotations.

## Quantified statements

The values `45%`, `35%`, and `22%` are not supported by the original repository.
They must not be presented as measured results until `scripts/benchmark.py`
produces an artifact containing:

1. dataset version and sample count;
2. baseline and candidate configuration;
3. metric formula;
4. raw per-question outcomes;
5. aggregate result and confidence/variance information.

Until then, use implementation-focused wording without percentages.
