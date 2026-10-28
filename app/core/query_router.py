"""Route conversational turns to local knowledge and/or live web retrieval."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalRoute:
    use_knowledge_base: bool = True
    use_web: bool = False
    reason: str = "knowledge"


class QueryRouter:
    """Deterministic, inspectable router for retrieval policy.

    This is intentionally separate from ranking. It prevents ordinary
    conversation from paying web-search latency while routing time-sensitive
    requests to live sources.
    """

    _CURRENT_PATTERNS = (
        r"\b(latest|recent|current|today|yesterday|now|news)\b",
        r"\b(weather|price|score|election|release date|version)\b",
        r"\b(last week|this month|this year)\b",
    )
    _SEARCH_PATTERNS = (
        r"\b(search|find|look up|lookup|web|online)\b",
        r"\baccording to\b",
    )
    _QUESTION_PATTERN = re.compile(
        r"^(what|how|who|when|where|why|tell me about)\b",
        re.IGNORECASE,
    )
    _GREETING_PATTERN = re.compile(
        r"^(hello|hi|hey|thanks|thank you|how are you)[?!. ]*$",
        re.IGNORECASE,
    )

    def route(self, message: str) -> RetrievalRoute:
        text = message.strip()
        if not text or self._GREETING_PATTERN.match(text):
            return RetrievalRoute(
                use_knowledge_base=False,
                use_web=False,
                reason="conversation",
            )
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self._CURRENT_PATTERNS):
            return RetrievalRoute(use_web=True, reason="time_sensitive")
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self._SEARCH_PATTERNS):
            return RetrievalRoute(use_web=True, reason="explicit_web")
        if self._QUESTION_PATTERN.search(text):
            return RetrievalRoute(use_web=True, reason="open_question")
        return RetrievalRoute(reason="knowledge")
