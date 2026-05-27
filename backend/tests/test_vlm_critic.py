"""Phase 2 VLM critic tests — uses HTML context instead of PainterPlan."""

from app.agents.vlm_critic import HeuristicVLMCritic
from app.core.errors import LLMCallError
from app.schemas.agents import ContentPlan, CritiqueResult, ElementContent
from app.schemas.layout import ElementType
from app.schemas.state import GraphState, RenderResult


class FakeVisionClient:
    """Test double that implements ``parse_vision``."""

    api_key = "key"
    base_url = "https://example.test/v1"
    model = "vision-model"

    def __init__(self, output, reasoning: str = ""):
        self.output = output  # CritiqueResult or Exception
        self.reasoning = reasoning
        self.calls = 0
        self.last_enable_thinking: bool | None = None

    async def parse_vision(
        self, *, messages, response_model, enable_thinking=True, thinking_budget=8192
    ):
        self.calls += 1
        self.last_enable_thinking = enable_thinking
        if isinstance(self.output, Exception):
            raise self.output
        user_content = messages[1]["content"]
        has_image = any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in user_content
        )
        assert has_image, "Vision call must include an image"
        return self.output, self.reasoning


def _state_for_critic() -> GraphState:
    state = GraphState(user_prompt="poster")
    state.content_plan = ContentPlan(
        poster_goal="test",
        elements=[
            ElementContent(id="title", type=ElementType.text, content="Title", priority=10)
        ],
    )
    state.layout_html = "<!DOCTYPE html><html><body><h1>Title</h1></body></html>"
    state.render_result = RenderResult(image_base64="iVBORw0KGgo=", width=512, height=768)
    return state


async def test_vlm_critic_uses_vision_client():
    output = CritiqueResult(
        score=88, passed=True, reasoning="ok",
        vision_description="A dark poster with centered white text",
    )
    reasoning = "Looking at the poster, I notice a clean layout..."
    client = FakeVisionClient(output, reasoning=reasoning)

    state = _state_for_critic()
    result = await HeuristicVLMCritic(vision_client=client).run(state)

    assert client.calls == 1
    assert client.last_enable_thinking is True
    assert result.score == 88
    assert state.vision_reasoning == reasoning


async def test_vlm_critic_falls_back_to_heuristic():
    client = FakeVisionClient(LLMCallError("vision unavailable"))
    state = _state_for_critic()
    result = await HeuristicVLMCritic(vision_client=client).run(state)

    assert client.calls == 1
    assert result.score >= 60
    assert result.vision_description == "(heuristic — no vision model available)"
    assert state.warnings


async def test_vlm_critic_raises_when_fallback_disabled():
    client = FakeVisionClient(LLMCallError("vision unavailable"))
    state = _state_for_critic()
    critic = HeuristicVLMCritic(vision_client=client)
    critic.allow_model_fallback = False
    import pytest
    with pytest.raises(LLMCallError):
        await critic.run(state)


async def test_vlm_critic_stores_empty_reasoning_when_thinking_off():
    output = CritiqueResult(score=80, passed=True, reasoning="ok")
    client = FakeVisionClient(output, reasoning="")
    critic = HeuristicVLMCritic(vision_client=client)
    critic.enable_thinking = False

    state = _state_for_critic()
    result = await critic.run(state)
    assert client.last_enable_thinking is False
    assert state.vision_reasoning == ""
    assert result.score == 80


def test_critique_result_normalizes_rubric_in_score_field():
    result = CritiqueResult.model_validate(
        {
            "score": {
                "poster_identity": 20,
                "topic_fit": 20,
                "composition": 18,
                "typography": 18,
                "readability": 20,
                "craft": 18,
            },
            "passed": True,
            "reasoning": "ok",
            "vision_description": "poster",
            "issues": [],
            "suggestions": [],
            "revision_focus": "final",
        }
    )

    assert result.score == 95
    assert result.rubric is not None
    assert result.rubric.composition == 18


async def test_vlm_critic_heuristic_requires_content_plan():
    """Heuristic needs content_plan with elements."""
    state = _state_for_critic()
    state.content_plan = None
    client = FakeVisionClient(LLMCallError("down"))
    import pytest
    with pytest.raises(ValueError, match="content_plan"):
        await HeuristicVLMCritic(vision_client=client).run(state)
