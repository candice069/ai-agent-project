import json

from app.core.feedback import FeedbackStore


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.operations = []

    def setex(self, key, ttl, value):
        self.operations.append(("setex", key, ttl, value))
        return self

    def zadd(self, key, mapping):
        self.operations.append(("zadd", key, mapping))
        return self

    def expire(self, key, ttl):
        self.operations.append(("expire", key, ttl))
        return self

    def execute(self):
        for operation in self.operations:
            if operation[0] == "setex":
                _, key, _, value = operation
                self.client.values[key] = value
        return [True] * len(self.operations)


class FakeRedis:
    def __init__(self):
        self.values = {}

    def setex(self, key, ttl, value):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)

    def pipeline(self):
        return FakePipeline(self)


def test_feedback_is_linked_to_prompt_run():
    redis_client = FakeRedis()
    store = FeedbackStore(redis_client=redis_client)
    store.record_response(
        message_id="message-1",
        session_id="session-1",
        question="Question",
        response="Answer",
        sources=[],
        prompt_metadata={"prompt_name": "rag", "prompt_version": "v1"},
    )

    feedback = store.submit_feedback(
        message_id="message-1",
        session_id="session-1",
        rating=-1,
        reason="Missing detail",
        corrected_answer="Better answer",
    )

    assert feedback["prompt"]["prompt_name"] == "rag"
    assert feedback["rating"] == -1
    saved = json.loads(redis_client.get("chat_feedback:message-1"))
    assert saved["corrected_answer"] == "Better answer"


def test_feedback_rejects_unknown_or_wrong_session():
    store = FeedbackStore(redis_client=FakeRedis())
    assert store.submit_feedback(
        message_id="missing",
        session_id="session",
        rating=1,
    ) is None

    store.record_response(
        message_id="message-1",
        session_id="session-1",
        question="Q",
        response="A",
        sources=[],
        prompt_metadata={},
    )
    assert store.submit_feedback(
        message_id="message-1",
        session_id="session-2",
        rating=1,
    ) is None
