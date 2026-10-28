from app.core.context_builder import ContextBuilder, estimate_tokens


def test_context_builder_respects_input_budget():
    builder = ContextBuilder(
        context_window=2400,
        response_budget=400,
        history_budget=500,
        retrieval_budget=500,
        summary_budget=200,
    )
    result = builder.build(
        user_input="question",
        history="old message\n" * 1000 + "newest message",
        summary="summary " * 1000,
        results=[
            {"title": f"Document {index}", "content": "content " * 1000}
            for index in range(3)
        ],
    )

    assert result.estimated_input_tokens <= 2000
    assert "newest message" in result.history
    assert estimate_tokens(result.context) <= 500
    assert estimate_tokens(result.summary) <= 200


def test_context_builder_keeps_ranked_sources_in_order():
    result = ContextBuilder().build(
        user_input="question",
        history="",
        summary="",
        results=[
            {"title": "First", "content": "one"},
            {"title": "Second", "content": "two"},
        ],
    )
    assert result.context.index("First") < result.context.index("Second")
