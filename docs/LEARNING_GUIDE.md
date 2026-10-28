# Module-by-Module Learning Guide

Use this order because every stage depends only on stages above it. For each
module, write a one-page note, build a minimal demo, run its tests, and practice
a two-minute explanation.

## Week 1: API, configuration, and LLM

1. Read `app/main.py`, `app/config.py`, `app/routes/chat.py`, and
   `app/services/llm_service.py`.
2. Rebuild a `/chat/` endpoint without retrieval.
3. Explain FastAPI dependency injection, async I/O, Pydantic validation, and why
   the API process should be stateless.
4. Validate with the API unit tests and Swagger at `/docs`.

## Week 2: Redis memory and Qdrant RAG

1. Read `app/core/memory.py`, `app/utils/embeddings.py`,
   `app/services/qdrant_service.py`, and `scripts/seed_knowledge.py`.
2. Start Redis and Qdrant, ingest two small documents, and inspect their payloads.
3. Verify two turns with the same session share context and different sessions do
   not.
4. Be able to calculate cosine similarity and explain embedding dimensions,
   chunk overlap, metadata filters, top-k, Recall@k, and MRR.

## Week 3: orchestration, web, and voice

1. Trace `AIAgent.process_message` from query embedding through citations.
2. Compare dense-only retrieval with hybrid fusion.
3. Exercise the SerpAPI route decision and provider failure fallback.
4. Run one STT request, one TTS request, and one text WebSocket turn.
5. Explain event-loop blocking, backpressure, provider latency, and graceful
   degradation.

## Week 4: realtime multimodal

1. Send WebSocket `text` and base64 `audio` events.
2. Observe `transcript`, `response`, and optional `audio` responses.
3. Measure stage latency and test disconnect/reconnect with the same session ID.
4. Compare REST request/response semantics with a long-lived WebSocket.

## Week 5: production memory and retrieval

1. Tune recent-message window, summary threshold, TTL, and summary size.
2. Compare recursive and semantic chunk boundaries.
3. Add a metadata-filtered question and inspect dense, lexical, fused, and final
   rankings.
4. Explain why deduplication belongs before context construction and why a
   cross-encoder reranker trades latency for precision.

## Week 6: evaluation and interview preparation

1. Version a fixed JSON question set.
2. Run baseline and candidate benchmarks into separate JSON files.
3. Compare latency, success, Recall@k, MRR, factuality, hallucination, coherence,
   and prompt reduction.
4. For each resume bullet, point to the entry file, main class, one test, one
   failure mode, and one measured artifact.

## Completion checklist

- A fresh clone starts from documented commands.
- Unit tests do not need vendor credentials.
- Integration tests explicitly opt into Redis/Qdrant.
- Both text and voice flows work end to end.
- Every percentage on the resume can be recomputed from checked-in evaluation
  inputs and generated result files.
