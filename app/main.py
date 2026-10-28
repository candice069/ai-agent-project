from fastapi import FastAPI
from fastapi.responses import JSONResponse
import asyncio
import redis
from qdrant_client import QdrantClient
from app.routes import chat, voice
from app.config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_PASSWORD,
    REDIS_DB,
    QDRANT_HOST,
    QDRANT_PORT,
)

app = FastAPI(
    title="AI Agent API",
    description="Humanized AI agent with natural language interaction in both text and voice",
    version="1.0.0"
)

# Include routers
app.include_router(chat.router)
app.include_router(voice.router)

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message"""
    return {"message": "Welcome to the Humanized AI Agent API"}

@app.get("/health")
async def health():
    """Report API, Redis, and Qdrant readiness."""
    components = {"api": "ok", "redis": "unavailable", "qdrant": "unavailable"}
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
        )
        await asyncio.to_thread(redis_client.ping)
        components["redis"] = "ok"
    except Exception:
        pass
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        await asyncio.to_thread(qdrant.get_collections)
        components["qdrant"] = "ok"
    except Exception:
        pass
    healthy = all(value == "ok" for value in components.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "components": components},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
