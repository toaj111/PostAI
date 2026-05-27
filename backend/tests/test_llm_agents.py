from app.agents.content_extractor import ContentExtractor
from app.agents.layout_planner import SpatialLayoutPlanner
from app.agents.style_director import StyleDirector
from app.core.errors import LLMCallError, SchemaParseError
from app.schemas.agents import (
    ArtDirectionV2,
    ColorSystem,
    ContentPlan,
    ContentStrategy,
    CritiqueResult,
    ElementContent,
    ImagerySpec,
    PosterBriefV2,
    PosterIntent,
    PosterLanguage,
    PosterMessage,
    StyleGuide,
    TypographySpec,
    VisualSubject,
)
from app.schemas.layout import CanvasSpec, ElementType
from app.schemas.state import GraphState, ReferenceImage


def _make_brief(headline: str = "发布会", subhead: str = "未来已来",
                cta: str | None = "报名", goal: str = "launch") -> PosterBriefV2:
    """Quickly build a PosterBriefV2 for test fixtures."""
    messages = [
        PosterMessage(id="title", role="headline", content=headline, importance=10, presence="required", source="user"),
    ]
    if subhead:
        messages.append(
            PosterMessage(id="subtitle", role="subhead", content=subhead, importance=8, presence="recommended", source="inferred"),
        )
    if cta:
        messages.append(
            PosterMessage(id="cta", role="cta", content=cta, importance=5, presence="required", source="inferred"),
        )
    return PosterBriefV2(
        poster_intent=PosterIntent(poster_type="campaign", communication_mode="persuade", primary_goal=goal),
        content_strategy=ContentStrategy(cta_policy="required" if cta else "omit"),
        messages=messages,
        visual_subjects=[
            VisualSubject(id="main_visual", role="illustration", description="robot", presence="recommended", source="inferred"),
        ],
    )


class FakeLLMClient:
    """Test double for StructuredLLMClient with controllable outputs."""
    api_key = "key"
    base_url = "https://example.test/v1"
    model = "real-model"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.captured_messages = []

    async def parse(self, *, messages, response_model):
        self.calls += 1
        self.captured_messages.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output

    async def _chat_completion(self, *, messages, response_model, force_raw=False):
        """Used by Phase 2 LayoutPlanner for raw HTML output."""
        self.calls += 1
        self.captured_messages.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class FakeMalformedContentClient:
    api_key = "key"
    base_url = "https://example.test/v1"
    model = "real-model"

    async def parse(self, *, messages, response_model):
        from app.core.errors import SchemaParseError
        raise SchemaParseError("bad schema")

    async def _chat_completion(self, *, messages, response_model, force_raw=False):
        import json
        return json.dumps(
            {
                "elements": [
                    {"id": "background", "type": "rect", "props": {"fill": "#0a0a2e"}},
                    {"id": "title", "type": "text", "props": {"text": "AI 会议"}},
                    {"id": "main_visual", "type": "image", "props": {"prompt": "AI visual"}},
                    {"id": "cta", "type": "group", "children": [{"type": "text", "text": "立即报名"}]},
                ]
            },
            ensure_ascii=False,
        )

    def validate_payload(self, payload, response_model):
        return response_model.model_validate(payload)


# ── ContentExtractor ──

async def test_content_extractor_uses_llm_output():
    brief = _make_brief(headline="发布会", subhead="未来已来", cta="报名", goal="launch")
    llm = FakeLLMClient([brief])
    state = GraphState(user_prompt="AI 发布会")
    result = await ContentExtractor(llm_client=llm).run(state)
    assert llm.calls == 1
    assert result.poster_goal == "launch"
    # Phase 2: poster_brief must be stored on state.
    assert state.poster_brief is not None
    assert state.poster_brief.poster_intent.primary_goal == "launch"
    assert state.poster_brief.content_strategy.cta_policy == "required"


async def test_content_extractor_normalizes_layout_like_payload():
    state = GraphState(user_prompt="AI 会议")
    result = await ContentExtractor(llm_client=FakeMalformedContentClient()).run(state)
    ids = {element.id for element in result.elements}
    assert {"title", "main_visual"} <= ids
    cta_element = next((e for e in result.elements if e.id == "cta"), None)
    assert cta_element is not None
    assert result.poster_goal
    # Phase 2: old-format payload is normalised into PosterBriefV2.
    assert state.poster_brief is not None


async def test_content_extractor_includes_reference_images_in_prompt():
    brief = _make_brief(headline="发布会", subhead="未来已来", cta="报名", goal="launch")
    llm = FakeLLMClient([brief])
    state = GraphState(
        user_prompt="AI 发布会",
        reference_images=[
            ReferenceImage(
                url="https://images.unsplash.com/photo-1518770660439-4636190af475",
                description="蓝色科技人物",
            )
        ],
    )
    await ContentExtractor(llm_client=llm).run(state)
    system_prompt = llm.captured_messages[0][0]["content"]
    assert "Reference images" in system_prompt or "Reference images" in llm.captured_messages[0][1]["content"]
    user_prompt = llm.captured_messages[0][1]["content"]
    assert "蓝色科技人物" in user_prompt


def _make_art_direction() -> ArtDirectionV2:
    """Quickly build an ArtDirectionV2 for test fixtures."""
    return ArtDirectionV2(
        style_name="tech neon",
        mood_keywords=["tech", "futuristic"],
        poster_language=PosterLanguage(composition_family="centered_iconic", visual_density="medium", negative_space="balanced", depth_strategy="layered", risk_level="safe"),
        color_system=ColorSystem(background="#000000", foreground="#FFFFFF", accent="#00E5FF", secondary="#111111", palette_notes="tech look"),
        typography=TypographySpec(headline_style="grotesk", body_style="sans", scale_contrast="high", letter_case="as_given"),
        imagery=ImagerySpec(treatment="none", background_strategy="gradient", prompt="clean", negative_prompt="clutter"),
    )


# ── StyleDirector ──

async def test_style_director_falls_back_when_llm_fails():
    llm = FakeLLMClient([LLMCallError("network down")])
    state = GraphState(user_prompt="科技海报")
    result = await StyleDirector(llm_client=llm).run(state)
    assert result.mood == "futuristic"
    assert state.warnings
    # Phase 3: art_direction stored even on fallback path.
    assert state.art_direction is not None


async def test_style_director_raises_when_fallback_disabled():
    llm = FakeLLMClient([LLMCallError("network down")])
    agent = StyleDirector(llm_client=llm)
    agent.allow_model_fallback = False
    state = GraphState(user_prompt="科技海报")
    import pytest
    with pytest.raises(LLMCallError):
        await agent.run(state)


async def test_style_director_includes_reference_images_in_prompt():
    llm = FakeLLMClient([_make_art_direction()])
    state = GraphState(
        user_prompt="科技海报",
        reference_images=[
            ReferenceImage(
                url="https://images.unsplash.com/photo-1461749280684-dccba630e2f6",
                description="深色背景与屏幕光效",
            )
        ],
    )
    state.content_plan = ContentPlan(
        poster_goal="test",
        elements=[ElementContent(id="title", type=ElementType.text, content="科技大会", priority=10)],
    )
    await StyleDirector(llm_client=llm).run(state)
    system_prompt = llm.captured_messages[0][0]["content"]
    assert "Reference images are provided" in system_prompt
    # Phase 3: art_direction stored on state.
    assert state.art_direction is not None
    assert state.art_direction.color_system.accent == "#00E5FF"
    assert "深色背景与屏幕光效" in system_prompt


# ── SpatialLayoutPlanner (Phase 2 — outputs HTML string) ──

_SAMPLE_HTML = """<!DOCTYPE html>
<html><head><style>
body { width:512px; height:768px; background:linear-gradient(180deg, #000, #111); }
.title { color:#FFF; font-size:40px; font-weight:bold; }
</style></head><body><div class="title">发布会</div></body></html>"""


def _state_for_planner() -> GraphState:
    state = GraphState(user_prompt="AI 发布会", canvas=CanvasSpec(width=512, height=768))
    state.content_plan = ContentPlan(
        poster_goal="launch",
        elements=[
            ElementContent(id="title", type=ElementType.text, content="发布会", priority=10),
            ElementContent(id="subtitle", type=ElementType.text, content="未来已来", priority=8),
            ElementContent(id="main_visual", type=ElementType.image, content="robot", priority=7),
            ElementContent(id="cta", type=ElementType.text, content="报名", priority=5),
        ],
    )
    state.style = StyleGuide(
        theme_keywords=["tech"],
        background_prompt="clean",
        primary_color="#000000",
        secondary_color="#111111",
        accent_color="#00E5FF",
        text_color="#FFFFFF",
        mood="futuristic",
    )
    return state


async def test_layout_planner_outputs_html_string():
    """Phase 2: LLM returns an HTML document string."""
    llm = FakeLLMClient([_SAMPLE_HTML])
    result = await SpatialLayoutPlanner(llm_client=llm).run(_state_for_planner())
    assert llm.calls == 1
    assert isinstance(result, str)
    assert "<!DOCTYPE html>" in result
    assert "发布会" in result


async def test_layout_planner_calls_llm_every_iteration():
    """Each run() calls the LLM again — no cached plan reuse."""
    html1 = _SAMPLE_HTML
    html2 = _SAMPLE_HTML.replace("40px", "60px")  # different
    llm = FakeLLMClient([html1, html2])
    planner = SpatialLayoutPlanner(llm_client=llm)
    state = _state_for_planner()

    result1 = await planner.run(state)
    assert llm.calls == 1
    assert "40px" in result1

    state.feedback_history.append(
        CritiqueResult(score=70, passed=False, reasoning="needs work", issues=["too tight top margin"])
    )

    result2 = await planner.run(state)
    assert llm.calls == 2, "Second iteration must call LLM again"
    assert "60px" in result2


async def test_layout_planner_falls_back_when_llm_fails():
    llm = FakeLLMClient([LLMCallError("network down")])
    state = _state_for_planner()
    result = await SpatialLayoutPlanner(llm_client=llm).run(state)
    assert isinstance(result, str)
    assert "<!DOCTYPE html>" in result
    assert state.warnings


async def test_layout_planner_raises_when_fallback_disabled():
    llm = FakeLLMClient([LLMCallError("network down")])
    planner = SpatialLayoutPlanner(llm_client=llm)
    planner.allow_model_fallback = False
    import pytest
    with pytest.raises(LLMCallError):
        await planner.run(_state_for_planner())


async def test_layout_planner_strips_markdown_fences():
    """When LLM wraps HTML in ```html fences, extract it."""
    fenced = '```html\n' + _SAMPLE_HTML + '\n```'
    llm = FakeLLMClient([fenced])
    result = await SpatialLayoutPlanner(llm_client=llm).run(_state_for_planner())
    assert not result.startswith("```")
    assert "<!DOCTYPE html>" in result


# ── Phase 3: HTML validation ──


def test_validate_html_accepts_valid_html():
    """_validate_html passes for reasonable HTML documents."""
    planner = SpatialLayoutPlanner()
    # Should not raise.
    planner._validate_html(_SAMPLE_HTML)
    planner._validate_html("<html><body><h1>Hello World, This is a Poster Title</h1></body></html>")


def test_validate_html_rejects_missing_tags():
    """_validate_html rejects content that lacks <body> or <html>."""
    import pytest
    planner = SpatialLayoutPlanner()
    with pytest.raises(SchemaParseError, match="body"):
        planner._validate_html("just some text, no html tags")


def test_validate_html_rejects_too_short():
    """_validate_html rejects content under 50 chars even with body tag."""
    import pytest
    planner = SpatialLayoutPlanner()
    with pytest.raises(SchemaParseError, match="too short"):
        planner._validate_html("<html><body><h1>Hi</h1></body></html>")


# ── Phase 3: feedback injection ──


async def test_layout_planner_includes_feedback_in_prompt():
    """When state has feedback_history, the LLM prompt includes VLM context."""
    planner = SpatialLayoutPlanner()
    state = _state_for_planner()
    state.feedback_history.append(
        CritiqueResult(
            score=65, passed=False, reasoning="spacing off",
            vision_description="Dark poster, white text at top",
            issues=["Title too close to edge"],
            suggestions=["Move title down by 0.05", "Add padding around text"],
        )
    )
    messages = planner._build_messages(state)
    user_content = messages[1]["content"]
    assert "FEEDBACK from VLM review" in user_content  # Phase 4 header
    assert "Title too close to edge" in user_content
    assert "Move title down" in user_content
    assert "What the vision model saw" in user_content
    assert "32px" not in user_content  # vision_reasoning not set


async def test_layout_planner_includes_vision_reasoning_in_feedback():
    """vision_reasoning from VLM is included in the feedback prompt."""
    planner = SpatialLayoutPlanner()
    state = _state_for_planner()
    state.vision_reasoning = "I see a dark poster with tight margins..."
    state.feedback_history.append(
        CritiqueResult(score=70, passed=False, reasoning="tight", issues=["margin too small"])
    )
    messages = planner._build_messages(state)
    user_content = messages[1]["content"]
    assert "Vision model reasoning:" in user_content
    assert "dark poster with tight margins" in user_content


def test_layout_planner_includes_reference_images_in_prompt():
    planner = SpatialLayoutPlanner()
    state = _state_for_planner()
    state.reference_images = [
        ReferenceImage(
            url="https://images.unsplash.com/photo-1518770660439-4636190af475",
            description="蓝色科技人物",
        )
    ]
    messages = planner._build_messages(state)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    # Phase 4: reference image guidance is in the system prompt.
    assert "reference image" in system_prompt.lower()
    # Phase 4: reference images appear in user prompt.
    assert "Reference images" in user_prompt
    assert "蓝色科技人物" in user_prompt
