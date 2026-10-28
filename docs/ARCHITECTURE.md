# AI Agent Architecture

## Module map

- `app/routes`: HTTP and WebSocket transport.
- `app/core/agent.py`: retrieval, ranking, prompting, and generation orchestration.
- `app/core/memory.py`: Redis-backed recent conversation history and summaries.
- `app/core/stt.py`, `app/core/tts.py`: pluggable speech providers.
- `app/services`: adapters for the LLM, Qdrant, Redis, and SerpAPI.
- `app/utils`: embeddings, retrieval ranking, deduplication, and response helpers.
- `scripts`: knowledge ingestion and reproducible evaluation.
- `tests`: unit tests; tests marked `integration` require Redis and Qdrant.

## Text request flow

```text
POST /chat/
  -> validate message and create/reuse session_id
  -> AIAgent.process_message
     -> load Redis conversation memory
     -> embed query
     -> hybrid Qdrant retrieval (dense + lexical)
     -> optional SerpAPI retrieval for current information
     -> deduplicate, fuse, filter, and rerank
     -> construct grounded prompt
     -> generate an LLM answer
     -> persist recent history and update the session summary
  -> return answer, citations, and session_id
```

## Voice and realtime flow

```text
POST /voice/          audio file -> STT -> agent -> JSON text response
POST /voice/stt       audio file -> STT -> transcript
POST /voice/tts       text -> TTS -> audio stream
WS   /voice/ws/{id}   audio/text events -> partial transcript -> agent -> text/audio events
```

The WebSocket protocol uses JSON events:

- client `{"type":"text","text":"..."}` sends a text turn;
- client `{"type":"audio","audio":"<base64>"}` sends a complete audio chunk;
- server emits `transcript`, `response`, optional `audio`, and `error` events.

## Runtime dependencies

- OpenAI-compatible LLM and embeddings
- Redis for session history and summaries
- Qdrant for vector and metadata storage
- SerpAPI for live search
- Whisper-compatible STT and Coqui XTTS, Google, Azure, or ElevenLabs TTS

## Design boundaries

The FastAPI app remains stateless. Durable state belongs in Redis and Qdrant, so
multiple API replicas can share sessions and knowledge. Vendor calls are isolated
behind services. Retrieval failures degrade to an ungrounded LLM turn; web search
failures do not fail the entire request.

## Primary failure modes

- Missing API keys: provider initialization or calls fail.
- Embedding dimension mismatch: Qdrant collection must be recreated when changing
  from a 1536-dimensional model to a 3072-dimensional model.
- Redis unavailable: durable conversation memory is unavailable.
- Empty Qdrant payload content: ingestion must store chunk text in each payload.
- Slow speech providers: WebSocket events expose stage boundaries so clients can
  show progress and measure transcription, generation, and synthesis latency.
