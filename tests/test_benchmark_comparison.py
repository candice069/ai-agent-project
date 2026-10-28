from scripts.compare_benchmarks import compare


def artifact(version, factuality, latency):
    return {
        "dataset_version": "1.0",
        "configuration": {"prompt_versions": {"rag": version}},
        "overall": {
            "num_questions": 10,
            "p95_time": latency,
            "quality": {
                "term_recall": factuality,
                "hallucination_rate": 0.1,
                "factuality_proxy": factuality,
                "recall_at_k": factuality,
            },
        },
    }


def test_candidate_is_eligible_only_when_all_gates_pass():
    result = compare(artifact("v1", 0.7, 1.0), artifact("v2", 0.8, 1.05))
    assert result["promotion"]["eligible"] is True
    assert result["promotion"]["candidate_prompt_versions"]["rag"] == "v2"

    regressed = compare(
        artifact("v1", 0.7, 1.0),
        artifact("v2", 0.6, 1.2),
    )
    assert regressed["promotion"]["eligible"] is False
