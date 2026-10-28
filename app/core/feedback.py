"""Redis persistence for prompt runs and reviewed user feedback."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import redis

from app.config import (
    MEMORY_TTL_SECONDS,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
)


class FeedbackStore:
    """Store prompt-run evidence and feedback with the session retention policy."""

    RESPONSE_PREFIX = "chat_response:"
    FEEDBACK_PREFIX = "chat_feedback:"
    FEEDBACK_INDEX = "chat_feedback:index"

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
        )

    def record_response(
        self,
        *,
        message_id: str,
        session_id: str,
        question: str,
        response: str,
        sources: Iterable[Dict[str, Any]],
        prompt_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        record = {
            "message_id": message_id,
            "session_id": session_id,
            "question": question,
            "response": response,
            "sources": list(sources),
            "prompt": prompt_metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.redis_client.setex(
            f"{self.RESPONSE_PREFIX}{message_id}",
            MEMORY_TTL_SECONDS,
            json.dumps(record),
        )
        return record

    def get_response(self, message_id: str) -> Optional[Dict[str, Any]]:
        payload = self.redis_client.get(f"{self.RESPONSE_PREFIX}{message_id}")
        return json.loads(payload) if payload else None

    def submit_feedback(
        self,
        *,
        message_id: str,
        session_id: str,
        rating: int,
        reason: Optional[str] = None,
        corrected_answer: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        response_record = self.get_response(message_id)
        if response_record is None:
            return None
        if response_record["session_id"] != session_id:
            return None

        now = datetime.now(timezone.utc)
        feedback = {
            **response_record,
            "rating": rating,
            "reason": reason,
            "corrected_answer": corrected_answer,
            "feedback_at": now.isoformat(),
        }
        key = f"{self.FEEDBACK_PREFIX}{message_id}"
        pipeline = self.redis_client.pipeline()
        pipeline.setex(key, MEMORY_TTL_SECONDS, json.dumps(feedback))
        pipeline.zadd(self.FEEDBACK_INDEX, {message_id: now.timestamp()})
        pipeline.expire(self.FEEDBACK_INDEX, MEMORY_TTL_SECONDS)
        pipeline.execute()
        return feedback
