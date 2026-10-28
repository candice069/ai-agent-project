#!/usr/bin/env python
"""Export and summarize prompt feedback for offline human review."""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.feedback import FeedbackStore


def load_feedback(store: FeedbackStore) -> List[Dict[str, Any]]:
    records = []
    for key in store.redis_client.scan_iter(
        match=f"{store.FEEDBACK_PREFIX}*"
    ):
        if key == store.FEEDBACK_INDEX:
            continue
        payload = store.redis_client.get(key)
        if payload:
            records.append(json.loads(payload))
    return records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_prompt = defaultdict(list)
    reasons = Counter()
    for record in records:
        prompt = record.get("prompt", {})
        prompt_key = (
            f"{prompt.get('prompt_name', 'unknown')}:"
            f"{prompt.get('prompt_version', 'unknown')}"
        )
        by_prompt[prompt_key].append(record.get("rating", 0))
        if record.get("reason"):
            reasons[record["reason"].strip().lower()] += 1

    prompt_summary = {}
    for prompt_key, ratings in sorted(by_prompt.items()):
        prompt_summary[prompt_key] = {
            "responses": len(ratings),
            "average_rating": sum(ratings) / len(ratings),
            "negative_ratings": sum(rating < 0 for rating in ratings),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feedback_count": len(records),
        "prompts": prompt_summary,
        "common_failure_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common()
        ],
        "negative_examples": [
            record for record in records if record.get("rating", 0) < 0
        ],
        "next_step": (
            "Review negative examples, create a new prompt version, and run "
            "the baseline/candidate benchmark before changing PROMPT_VERSION."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and summarize Redis-backed response feedback"
    )
    parser.add_argument(
        "--output",
        default="benchmark-results/feedback-analysis.json",
    )
    args = parser.parse_args()

    report = summarize(load_feedback(FeedbackStore()))
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
    print(f"Wrote {report['feedback_count']} feedback records to {args.output}")


if __name__ == "__main__":
    main()
