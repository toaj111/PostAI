from app.agents.content_expander import ContentExpander
from app.agents.content_extractor import ContentExtractor
from app.agents.layout_planner import SpatialLayoutPlanner
from app.agents.style_director import StyleDirector
from app.agents.visual_system_planner import VisualSystemPlanner
from app.core.errors import LLMCallError, SchemaParseError
from app.schemas.agents import (
    ArtDirectionV2,
    ColorSystem,
    ContentExpansionPlan,
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
    VisualLayerSpec,
    VisualSubject,
    VisualSystemPlan,
    poster_brief_to_content_plan,
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


# ── ContentExpander ──

async def test_content_expander_adds_music_festival_information_slots():
    state = GraphState(
        user_prompt="生成一张音乐节的海报",
        canvas=CanvasSpec(width=512, height=768),
    )
    state.poster_brief = PosterBriefV2(
        poster_intent=PosterIntent(
            poster_type="event",
            communication_mode="celebrate",
            primary_goal="music festival poster",
            tone=["energetic"],
        ),
        content_strategy=ContentStrategy(cta_policy="omit", information_density="medium"),
        messages=[
            PosterMessage(
                id="headline",
                role="headline",
                content="音乐节",
                importance=10,
                presence="required",
                source="user",
            )
        ],
        visual_subjects=[],
    )
    state.content_plan = poster_brief_to_content_plan(state.poster_brief)

    plan = await ContentExpander().run(state)

    assert plan.poster_type == "music_event"
    assert state.content_expansion is plan
    assert state.poster_brief is not None
    contents = {message.content for message in state.poster_brief.messages}
    assert {"DATE TBD", "VENUE TBD", "LINEUP TBA", "TICKETS INFO"} <= contents
    assert state.poster_brief.content_strategy.information_density == "dense"
    assert state.poster_brief.content_strategy.cta_policy == "optional"
    assert any(subject.id == "stage-light-system" for subject in state.poster_brief.visual_subjects)
    assert state.content_plan is not None
    assert any(element.content == "LINEUP TBA" for element in state.content_plan.elements)


async def test_content_expander_keeps_minimal_type_only_sparse():
    state = GraphState(user_prompt="做一张纯文字音乐节海报，极简排版，不要额外信息")
    state.poster_brief = PosterBriefV2(
        poster_intent=PosterIntent(
            poster_type="typographic",
            communication_mode="evoke",
            primary_goal="minimal music poster",
            tone=["minimal"],
        ),
        content_strategy=ContentStrategy(
            cta_policy="omit",
            image_policy="omit",
            information_density="sparse",
        ),
        messages=[
            PosterMessage(
                id="headline",
                role="headline",
                content="JAZZ NIGHT",
                importance=10,
                presence="required",
                source="user",
            )
        ],
    )
    state.content_plan = poster_brief_to_content_plan(state.poster_brief)

    plan = await ContentExpander().run(state)

    assert plan.density_recommendation == "sparse"
    assert not plan.inferred_messages
    assert state.poster_brief.content_strategy.information_density == "sparse"
    assert {message.content for message in state.poster_brief.messages} == {"JAZZ NIGHT"}


async def test_content_expander_uses_llm_output():
    expansion = ContentExpansionPlan(
        poster_type="event",
        density_recommendation="dense",
        self_questions=["What missing event slots are needed?"],
        inferred_messages=[
            PosterMessage(
                id="date_tbd",
                role="date",
                content="DATE TBD",
                importance=8,
                presence="recommended",
                source="placeholder",
            )
        ],
    )
    llm = FakeLLMClient([expansion])
    state = GraphState(user_prompt="活动海报")
    state.poster_brief = _make_brief(cta=None)
    state.content_plan = poster_brief_to_content_plan(state.poster_brief)

    result = await ContentExpander(llm_client=llm).run(state)

    assert llm.calls == 1
    assert result is expansion
    assert state.content_expansion is expansion
    assert any(message.content == "DATE TBD" for message in state.poster_brief.messages)


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


def _make_visual_system() -> VisualSystemPlan:
    return VisualSystemPlan(
        composition_archetype="diagonal_energy",
        density="dense",
        focal_strategy="headline and symbol pull across a diagonal axis",
        layer_count_target=7,
        required_html_ids=["base-field", "texture-field", "headline-system", "key-visual-system"],
        layers=[
            VisualLayerSpec(id="base-field", role="background", description="deep field", priority=10, presence="required"),
            VisualLayerSpec(id="texture-field", role="texture", description="fine grid texture", priority=8, presence="required"),
            VisualLayerSpec(id="headline-system", role="typography", description="large cropped headline", priority=10, presence="required"),
            VisualLayerSpec(id="key-visual-system", role="symbol", description="large abstract AI node symbol", priority=8, presence="required"),
        ],
        typography_treatment="bold grotesk, high contrast",
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


# ── VisualSystemPlanner ──

async def test_visual_system_planner_uses_llm_output():
    plan = _make_visual_system()
    llm = FakeLLMClient([plan])
    state = _state_for_planner()
    state.poster_brief = _make_brief()
    state.art_direction = _make_art_direction()

    result = await VisualSystemPlanner(llm_client=llm).run(state)

    assert llm.calls == 1
    assert result.composition_archetype == "diagonal_energy"
    assert "texture-field" in result.required_html_ids
    assert state.visual_system is result
    user_prompt = llm.captured_messages[0][1]["content"]
    assert "PosterBriefV2" in user_prompt
    assert "ArtDirectionV2" in user_prompt


async def test_visual_system_planner_falls_back_when_llm_fails():
    llm = FakeLLMClient([LLMCallError("network down")])
    state = _state_for_planner()
    state.poster_brief = _make_brief()
    state.art_direction = _make_art_direction()

    result = await VisualSystemPlanner(llm_client=llm).run(state)

    assert state.warnings
    assert state.visual_system is result
    assert result.layer_count_target >= 5
    assert any(layer.id == "texture-field" for layer in result.layers)
    assert "headline-system" in result.required_html_ids


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


def test_layout_planner_fallback_keeps_dense_recruitment_details():
    state = _state_for_planner()
    state.poster_brief = PosterBriefV2(
        poster_intent=PosterIntent(
            poster_type="recruitment",
            communication_mode="recruit",
            primary_goal="招聘人工智能算法工程师",
        ),
        content_strategy=ContentStrategy(
            information_density="dense",
            cta_policy="required",
            image_policy="optional",
        ),
        messages=[
            PosterMessage(id="headline", role="headline", content="字节跳动AI算法工程师招聘", importance=10, presence="required", source="user"),
            PosterMessage(id="subhead", role="subhead", content="加入我们，用AI改变世界", importance=7, presence="recommended", source="inferred"),
            PosterMessage(id="salary", role="body", content="薪资：面议（竞争力薪资）", importance=8, presence="required", source="placeholder"),
            PosterMessage(id="benefits", role="body", content="福利：六险一金、免费三餐、股票期权等", importance=8, presence="required", source="placeholder"),
            PosterMessage(id="requirements", role="body", content="要求：扎实算法基础，熟悉深度学习框架", importance=6, presence="recommended", source="inferred"),
            PosterMessage(id="cta_qrcode", role="cta", content="扫码报名", importance=10, presence="required", source="user"),
        ],
    )
    html = SpatialLayoutPlanner()._run_fallback(state)
    assert "薪资：面议" in html
    assert "福利：六险一金" in html
    assert "要求：扎实算法基础" in html
    assert 'id="qr-code"' in html
    assert "clear action, strong hierarchy" not in html


def test_layout_planner_fallback_keeps_music_festival_expanded_details():
    state = _state_for_planner()
    state.user_prompt = "音乐节演出海报，高能量、大胆构图、霓虹色彩"
    state.poster_brief = PosterBriefV2(
        poster_intent=PosterIntent(
            poster_type="event",
            communication_mode="celebrate",
            primary_goal="music festival poster",
        ),
        content_strategy=ContentStrategy(
            information_density="dense",
            cta_policy="optional",
            image_policy="optional",
        ),
        messages=[
            PosterMessage(id="headline", role="headline", content="NEON BEATS", importance=10, presence="required", source="user"),
            PosterMessage(id="date_tbd", role="date", content="DATE TBD", importance=8, presence="recommended", source="placeholder"),
            PosterMessage(id="venue_tbd", role="venue", content="VENUE TBD", importance=8, presence="recommended", source="placeholder"),
            PosterMessage(id="lineup_tba", role="body", content="LINEUP TBA", importance=8, presence="recommended", source="placeholder"),
            PosterMessage(id="stage_program", role="meta", content="MAIN STAGE / BASS STAGE / DANCE STAGE", importance=6, presence="recommended", source="inferred"),
            PosterMessage(id="ticket_info", role="cta", content="TICKETS INFO", importance=6, presence="optional", source="placeholder"),
        ],
        visual_subjects=[
            VisualSubject(id="stage-light-system", role="symbol", description="stage light beams", presence="recommended", source="inferred"),
        ],
    )
    html = SpatialLayoutPlanner()._run_fallback(state)
    assert "DATE TBD" in html
    assert "VENUE TBD" in html
    assert "LINEUP TBA" in html
    assert "MAIN STAGE / BASS STAGE / DANCE STAGE" in html
    assert "TICKETS INFO" in html
    assert 'id="qr-code"' not in html
    assert "POSTER DETAILS" in html


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


def test_layout_planner_includes_visual_system_in_prompt():
    planner = SpatialLayoutPlanner()
    state = _state_for_planner()
    state.visual_system = _make_visual_system()
    messages = planner._build_messages(state)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "VISUAL SYSTEM PLAN" in system_prompt
    assert "VisualSystemPlan" in user_prompt
    assert "key-visual-system" in user_prompt


def test_layout_planner_includes_content_expansion_in_prompt():
    planner = SpatialLayoutPlanner()
    state = _state_for_planner()
    state.content_expansion = ContentExpansionPlan(
        poster_type="music_event",
        density_recommendation="dense",
        self_questions=["What should a music festival poster include?"],
        inferred_messages=[
            PosterMessage(
                id="lineup_tba",
                role="body",
                content="LINEUP TBA",
                importance=8,
                presence="recommended",
                source="placeholder",
            )
        ],
    )
    messages = planner._build_messages(state)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "ContentExpansionPlan placeholders" in system_prompt
    assert "ContentExpansionPlan" in user_prompt
    assert "LINEUP TBA" in user_prompt
