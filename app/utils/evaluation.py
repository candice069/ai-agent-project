"""Deterministic evaluation helpers for repeatable agent benchmarks."""

from statistics import mean
from typing import Any, Dict, Iterable, List


def score_answer(
    response: str,
    sources: List[Dict[str, Any]],
    case: Dict[str, Any],
) -> Dict[str, float]:
    """Score one answer against explicit labels in an evaluation case."""
    response_lower = response.lower()
    expected_terms = [str(term).lower() for term in case.get("expected_terms", [])]
    forbidden_terms = [str(term).lower() for term in case.get("forbidden_terms", [])]
    expected_sources = [
        str(source).lower() for source in case.get("expected_sources", [])
    ]

    term_recall = (
        sum(term in response_lower for term in expected_terms) / len(expected_terms)
        if expected_terms
        else 1.0
    )
    hallucination_rate = (
        sum(term in response_lower for term in forbidden_terms) / len(forbidden_terms)
        if forbidden_terms
        else 0.0
    )

    source_texts = [
        f"{source.get('title', '')} {source.get('url', '')}".lower()
        for source in sources
    ]
    matched_sources = [
        expected
        for expected in expected_sources
        if any(expected in actual for actual in source_texts)
    ]
    recall_at_k = (
        len(matched_sources) / len(expected_sources) if expected_sources else 1.0
    )
    reciprocal_rank = 0.0
    if expected_sources:
        for rank, actual in enumerate(source_texts, start=1):
            if any(expected in actual for expected in expected_sources):
                reciprocal_rank = 1.0 / rank
                break
    else:
        reciprocal_rank = 1.0

    return {
        "term_recall": term_recall,
        "hallucination_rate": hallucination_rate,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "factuality_proxy": term_recall * (1.0 - hallucination_rate),
    }


def aggregate_scores(scores: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Average all metric keys across successful cases."""
    rows = list(scores)
    if not rows:
        return {}
    return {
        key: mean(row[key] for row in rows)
        for key in rows[0]
    }
