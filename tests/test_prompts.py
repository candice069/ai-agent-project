from app.core.prompts import PROMPTS, get_prompt
from app.core.query_router import QueryRouter


def test_all_prompts_render_with_expected_inputs():
    common = {
        "history": "Human: hello",
        "summary": "The user prefers concise answers.",
        "context": "Source: Guide\nA supported fact.",
        "input": "What is supported?",
    }
    for name in ("general", "rag", "web", "voice"):
        rendered = PROMPTS[name].template.format(**common)
        assert "What is supported?" in rendered
        assert PROMPTS[name].version


def test_grounded_prompts_include_few_shot_examples():
    values = {"history": "", "summary": "", "context": "", "input": "Question"}
    assert "Example human question" in PROMPTS["rag"].template.format(**values)
    assert "Example human question" in PROMPTS["web"].template.format(**values)
    assert "Example human question" not in PROMPTS["general"].template.format(
        **values
    )


def test_prompt_fallback_and_route_selection():
    router = QueryRouter()
    assert router.route("hello").prompt_type == "general"
    assert router.route("latest AI news").prompt_type == "web"
    assert router.route("Explain our architecture").prompt_type == "rag"
    assert get_prompt("missing").name == "general"
