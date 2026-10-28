"""Few-shot examples for prompt routes that require strict grounding."""

RAG_EXAMPLES = [
    {
        "question": "Where is session memory stored?",
        "context": "Source: Architecture guide\nConversation state is stored in Redis.",
        "answer": (
            "Session memory is stored in Redis. "
            "[Source: Architecture guide]"
        ),
    },
    {
        "question": "Which database stores user profiles?",
        "context": "Source: Architecture guide\nRedis stores conversation state.",
        "answer": (
            "The supplied context does not identify a database for user profiles. "
            "I don't know based on the available sources."
        ),
    },
]

WEB_EXAMPLES = [
    {
        "question": "What is the current release?",
        "context": (
            "Source: Official release notes\n"
            "Version 2.4 was released on 2026-08-10."
        ),
        "answer": (
            "The current release in the provided results is version 2.4, "
            "released on 2026-08-10. [Source: Official release notes]"
        ),
    }
]
