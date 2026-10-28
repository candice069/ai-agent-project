# AI Agent Knowledge Base

This sample document verifies the ingestion and retrieval pipeline.

The conversational agent uses FastAPI for transport, Redis for session memory,
Qdrant for vector retrieval, and SerpAPI for time-sensitive web grounding.
Hybrid retrieval combines dense semantic similarity with lexical token overlap.
The final context is deduplicated and ranked before it is sent to the language
model.
