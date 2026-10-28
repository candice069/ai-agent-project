#!/usr/bin/env python
"""Compare two benchmark artifacts without inventing resume metrics."""

import argparse
import json
from typing import Any, Dict


def percent_change(baseline: float, candidate: float, lower_is_better: bool = False) -> float:
    if baseline == 0:
        return 0.0
    change = (candidate - baseline) / abs(baseline) * 100.0
    return -change if lower_is_better else change


def compare(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    base_quality = baseline.get("overall", {}).get("quality", {})
    candidate_quality = candidate.get("overall", {}).get("quality", {})
    same_dataset = (
        baseline.get("dataset_version") == candidate.get("dataset_version")
        and baseline.get("overall", {}).get("num_questions")
        == candidate.get("overall", {}).get("num_questions")
    )
    quality_gates = {
        metric: candidate_quality.get(metric, 0.0)
        >= base_quality.get(metric, 0.0)
        for metric in ("term_recall", "factuality_proxy", "recall_at_k")
    }
    hallucination_gate = (
        candidate_quality.get("hallucination_rate", 0.0)
        <= base_quality.get("hallucination_rate", 0.0)
    )
    baseline_p95 = baseline.get("overall", {}).get("p95_time", 0.0)
    candidate_p95 = candidate.get("overall", {}).get("p95_time", 0.0)
    latency_gate = baseline_p95 == 0 or candidate_p95 <= baseline_p95 * 1.1
    gates = {
        "same_dataset": same_dataset,
        **quality_gates,
        "hallucination_rate": hallucination_gate,
        "p95_latency_within_10_percent": latency_gate,
    }
    return {
        "sample_count": candidate.get("overall", {}).get("num_questions", 0),
        "dataset_version": candidate.get("dataset_version", "unknown"),
        "improvements_percent": {
            "context_coherence_proxy": percent_change(
                base_quality.get("term_recall", 0.0),
                candidate_quality.get("term_recall", 0.0),
            ),
            "hallucination_reduction": percent_change(
                base_quality.get("hallucination_rate", 0.0),
                candidate_quality.get("hallucination_rate", 0.0),
                lower_is_better=True,
            ),
            "factuality_proxy": percent_change(
                base_quality.get("factuality_proxy", 0.0),
                candidate_quality.get("factuality_proxy", 0.0),
            ),
            "recall_at_k": percent_change(
                base_quality.get("recall_at_k", 0.0),
                candidate_quality.get("recall_at_k", 0.0),
            ),
            "latency_reduction": percent_change(
                baseline.get("overall", {}).get("p95_time", 0.0),
                candidate.get("overall", {}).get("p95_time", 0.0),
                lower_is_better=True,
            ),
        },
        "promotion": {
            "eligible": all(gates.values()),
            "gates": gates,
            "candidate_prompt_versions": candidate.get(
                "configuration", {}
            ).get("prompt_versions", {}),
            "note": (
                "Eligibility is an automated regression gate; a human must "
                "still review raw answers before changing PROMPT_VERSION."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.baseline, encoding="utf-8") as file:
        baseline = json.load(file)
    with open(args.candidate, encoding="utf-8") as file:
        candidate = json.load(file)
    result = compare(baseline, candidate)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(rendered + "\n")


if __name__ == "__main__":
    main()
