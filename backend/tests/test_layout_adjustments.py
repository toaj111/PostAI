"""Phase 2: verify that layout planner re-generates HTML on each call.

The old adjustment-vector approach is completely deleted.  Each iteration
calls the LLM and produces a fresh HTML document.
"""

from app.agents.layout_planner import SpatialLayoutPlanner
from app.schemas.agents import ContentPlan, CritiqueResult, ElementContent, StyleGuide
from app.schemas.layout import CanvasSpec, ElementType
from app.schemas.state import GraphState

_HTML1 = "<!DOCTYPE html><html><body style='width:512px;height:768px'><h1>Hello</h1></body></html>"
_HTML2 = "<!DOCTYPE html><html><body style='width:512px;height:768px'><h1>Hello</h1><p>V2</p></body></html>"


async def test_layout_planner_regenerates_on_each_call():
    """Two consecutive calls produce fresh HTML — no stale caching."""
    class TwoHTMLClient:
        api_key = "k"
        base_url = "https://x.test/v1"
        model = "m"
        calls = 0

        def __init__(self):
            self._htmls = [_HTML1, _HTML2]

        async def _chat_completion(self, *, messages, response_model, force_raw=False):
            self.calls += 1
            return self._htmls.pop(0)

    client = TwoHTMLClient()
    planner = SpatialLayoutPlanner(llm_client=client)

    state = GraphState(user_prompt="poster", canvas=CanvasSpec(width=512, height=768))
    state.content_plan = ContentPlan(
        poster_goal="test",
        elements=[ElementContent(id="title", type=ElementType.text, content="Hello", priority=10)],
    )
    state.style = StyleGuide(
        theme_keywords=["test"], background_prompt="clean",
        primary_color="#000", secondary_color="#111",
        accent_color="#0FF", text_color="#FFF", mood="futuristic",
    )

    result1 = await planner.run(state)
    assert client.calls == 1
    assert "V2" not in result1

    state.feedback_history.append(
        CritiqueResult(score=70, passed=False, reasoning="need space", issues=["tight top"])
    )

    result2 = await planner.run(state)
    assert client.calls == 2, "Should call LLM on every run"
    assert "V2" in result2


async def test_feedback_drives_html_changes():
    """Phase 3: when VLM feedback is present, the LLM prompt includes it,
    and the second output should differ from the first."""
    class FeedbackCapturingClient:
        api_key = "k"
        base_url = "https://x.test/v1"
        model = "m"
        calls = 0
        captured_messages: list = []

        def __init__(self):
            self._htmls = [
                "<!DOCTYPE html><html><body><h1>V1</h1><p>A</p></body></html>",
                "<!DOCTYPE html><html><body><h1>V1</h1><p>B</p></body></html>",
            ]

        async def _chat_completion(self, *, messages, response_model, force_raw=False):
            self.calls += 1
            self.captured_messages.append(messages)
            return self._htmls.pop(0)

    client = FeedbackCapturingClient()
    planner = SpatialLayoutPlanner(llm_client=client)

    state = GraphState(user_prompt="poster", canvas=CanvasSpec(width=512, height=768))
    state.content_plan = ContentPlan(
        poster_goal="test",
        elements=[ElementContent(id="title", type=ElementType.text, content="Title", priority=10)],
    )
    state.style = StyleGuide(
        theme_keywords=["test"], background_prompt="clean",
        primary_color="#000", secondary_color="#111",
        accent_color="#0FF", text_color="#FFF", mood="futuristic",
    )

    # First call — no feedback.
    result1 = await planner.run(state)
    assert client.calls == 1
    assert "<h1>V1</h1>" in result1
    first_prompt = client.captured_messages[0][1]["content"]
    assert "Feedback" not in first_prompt  # No feedback on first iteration.

    # Add feedback and iterate.
    state.feedback_history.append(
        CritiqueResult(
            score=65, passed=False, reasoning="needs improvement",
            vision_description="A poster with white title on dark background",
            issues=["Title font is too small", "Missing visual element"],
            suggestions=["Increase title font-size to 48px", "Add a decorative SVG below the title"],
        )
    )

    # Second call — should include feedback.
    result2 = await planner.run(state)
    assert client.calls == 2
    assert "<h1>V1</h1>" in result2
    assert "<p>B</p>" in result2  # Different from <p>A</p> in V1.
    # captured_messages[1] = list of messages for 2nd call; [1] = user message
    second_prompt = client.captured_messages[1][1]["content"]
    assert "FEEDBACK from VLM review" in second_prompt  # Phase 4 header
    assert "Title font is too small" in second_prompt
    assert "Increase title font-size to 48px" in second_prompt
