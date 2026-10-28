"""Token-aware assembly of conversation and retrieval context."""

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, Iterable

from app.config import (
    LLM_CONTEXT_WINDOW,
    MAX_HISTORY_TOKENS,
    MAX_RETRIEVAL_TOKENS,
    MAX_SUMMARY_TOKENS,
    RESPONSE_TOKEN_BUDGET,
)


def estimate_tokens(text: str) -> int:
    """Estimate tokens conservatively without adding a tokenizer dependency."""
    if not text:
        return 0
    return max(1, ceil(len(text) / 4))


def truncate_to_tokens(text: str, budget: int, *, keep_tail: bool = False) -> str:
    """Trim text to an approximate token budget."""
    if not text or budget <= 0:
        return ""
    max_chars = budget * 4
    if len(text) <= max_chars:
        return text
    marker = "\n[...truncated...]\n"
    remaining = max(max_chars - len(marker), 0)
    if keep_tail:
        return marker + text[-remaining:] if remaining else marker[:max_chars]
    return text[:remaining] + marker if remaining else marker[:max_chars]


@dataclass(frozen=True)
class PromptContext:
    history: str
    summary: str
    context: str
    estimated_input_tokens: int


class ContextBuilder:
    """Allocate bounded prompt space to summary, recent history, and sources."""

    def __init__(
        self,
        context_window: int = LLM_CONTEXT_WINDOW,
        response_budget: int = RESPONSE_TOKEN_BUDGET,
        history_budget: int = MAX_HISTORY_TOKENS,
        retrieval_budget: int = MAX_RETRIEVAL_TOKENS,
        summary_budget: int = MAX_SUMMARY_TOKENS,
    ):
        self.input_budget = max(1, context_window - response_budget)
        self.history_budget = max(0, history_budget)
        self.retrieval_budget = max(0, retrieval_budget)
        self.summary_budget = max(0, summary_budget)

    def build(
        self,
        *,
        user_input: str,
        history: str,
        summary: str,
        results: Iterable[Dict[str, Any]],
    ) -> PromptContext:
        # Keep room for instructions, few-shot examples, and formatting.
        available = max(
            0,
            self.input_budget - estimate_tokens(user_input) - 1000,
        )

        summary_limit = min(self.summary_budget, available)
        bounded_summary = truncate_to_tokens(str(summary or ""), summary_limit)
        available -= estimate_tokens(bounded_summary)

        history_limit = min(self.history_budget, available)
        # The newest messages are at the end of LangChain's rendered history.
        bounded_history = truncate_to_tokens(
            str(history or ""),
            history_limit,
            keep_tail=True,
        )
        available -= estimate_tokens(bounded_history)

        retrieval_limit = min(self.retrieval_budget, available)
        context_parts = []
        used = 0
        for result in results:
            document = (
                f"Source: {result.get('title', 'Unknown')}\n"
                f"{result.get('content', '')}"
            )
            remaining = retrieval_limit - used
            if remaining <= 0:
                break
            bounded_document = truncate_to_tokens(document, remaining)
            if bounded_document:
                context_parts.append(bounded_document)
                used += estimate_tokens(bounded_document)

        bounded_context = "\n\n".join(context_parts)
        estimated = (
            estimate_tokens(user_input)
            + estimate_tokens(bounded_summary)
            + estimate_tokens(bounded_history)
            + estimate_tokens(bounded_context)
            + 1000
        )
        return PromptContext(
            history=bounded_history,
            summary=bounded_summary,
            context=bounded_context,
            estimated_input_tokens=estimated,
        )
