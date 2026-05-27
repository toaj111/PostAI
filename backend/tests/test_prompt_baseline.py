"""Phase 0 baseline tests: document current prompt and CTA behaviour.

These tests capture the behaviour of every agent's deterministic fallback path
so that future prompt/schema changes have a clear regression suite.

Representative inputs (fixtures):
  - 纯文字爵士音乐节海报，不要按钮
  - 当代艺术展极简海报
  - 招聘海报，需要扫码报名
  - 信息密集的讲座日程海报
  - 只用抽象形状表达海边夏日
"""

import pytest

from app.agents.content_extractor import ContentExtractor
from app.agents.layout_planner import SpatialLayoutPlanner
from app.agents.style_director import StyleDirector
from app.agents.vlm_critic import HeuristicVLMCritic
from app.core.errors import LLMCallError, SchemaParseError
from app.schemas.agents import (
    ArtDirectionV2,
    ColorSystem,
    ContentPlan,
    ContentStrategy,
    ElementContent,
    ImagerySpec,
    PosterBriefV2,
    PosterIntent,
    PosterLanguage,
    PosterMessage,
    StyleGuide,
    TypographySpec,
)
from app.schemas.layout import CanvasSpec, ElementType
from app.schemas.state import GraphState, RenderResult


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0 — ContentExtractor baseline
# ═══════════════════════════════════════════════════════════════════════════════


class TestContentExtractorBaseline:
    """Document the ContentExtractor fallback behaviour for representative inputs."""

    def test_no_cta_for_art_poster(self):
        """纯文字爵士音乐节海报，不要按钮 → CTA should NOT be generated."""
        state = GraphState(user_prompt="做一张纯文字爵士音乐节海报，不要按钮")
        plan = ContentExtractor()._run_rules(state)
        ids = {e.id for e in plan.elements}
        assert "cta" not in ids, "CTA must not appear when user says '不要按钮'"
        assert "title" in ids

    def test_no_cta_for_minimal_art_exhibition(self):
        """当代艺术展极简海报 → no CTA needed."""
        state = GraphState(user_prompt="为一个当代艺术展做极简海报")
        plan = ContentExtractor()._run_rules(state)
        ids = {e.id for e in plan.elements}
        assert "cta" not in ids, "Minimal art exhibition poster should not have CTA"

    def test_cta_for_recruitment_with_qr(self):
        """招聘海报，需要扫码报名 → CTA SHOULD be generated."""
        state = GraphState(user_prompt="招聘海报，需要扫码报名")
        plan = ContentExtractor()._run_rules(state)
        ids = {e.id for e in plan.elements}
        assert "cta" in ids, "Recruitment poster with '扫码报名' must have CTA"
        cta = next(e for e in plan.elements if e.id == "cta")
        assert cta.presence == "required"
        assert cta.role == "cta"

    def test_no_cta_for_dense_info_poster(self):
        """信息密集的讲座日程海报 → no CTA unless signup keywords present."""
        state = GraphState(user_prompt="制作信息密集的讲座日程海报")
        plan = ContentExtractor()._run_rules(state)
        ids = {e.id for e in plan.elements}
        assert "cta" not in ids, "Lecture schedule poster doesn't need CTA by default"

    def test_no_cta_for_abstract_summer(self):
        """只用抽象形状表达海边夏日 → pure visual, no CTA."""
        state = GraphState(user_prompt="只用抽象形状表达海边夏日")
        plan = ContentExtractor()._run_rules(state)
        ids = {e.id for e in plan.elements}
        assert "cta" not in ids, "Abstract shape poster should not have CTA"

    def test_cta_for_purchase_intent(self):
        """购买 intent → CTA should be generated."""
        state = GraphState(user_prompt="限时购买特惠产品海报")
        plan = ContentExtractor()._run_rules(state)
        ids = {e.id for e in plan.elements}
        assert "cta" in ids
        assert any("购买" in e.content for e in plan.elements if e.id == "cta")

    def test_cta_for_register_intent(self):
        """注册/报名 intent → CTA with appropriate text."""
        state = GraphState(user_prompt="技术大会注册报名海报")
        plan = ContentExtractor()._run_rules(state)
        ids = {e.id for e in plan.elements}
        assert "cta" in ids
        assert any("报名" in e.content for e in plan.elements if e.id == "cta")

    def test_all_elements_have_role_and_presence(self):
        """Every element from the fallback must include role and presence fields."""
        state = GraphState(user_prompt="科技AI大会海报")
        plan = ContentExtractor()._run_rules(state)
        for el in plan.elements:
            assert el.presence in ("required", "recommended", "optional", "omit"), (
                f"Element {el.id} missing valid presence, got {el.presence!r}"
            )

    def test_validate_requires_at_least_one_core_element(self):
        """Validation rejects briefs with no importance>=8 required message."""
        brief = PosterBriefV2(
            poster_intent=PosterIntent(primary_goal="test"),
            messages=[
                PosterMessage(id="x", role="body", content="low", importance=3, presence="required", source="user"),
            ],
        )
        with pytest.raises(SchemaParseError, match="importance >= 8"):
            ContentExtractor()._validate_brief(brief)

    def test_validate_accepts_brief_with_core_headline(self):
        """Validation passes when at least one required, importance>=8 message exists."""
        brief = PosterBriefV2(
            poster_intent=PosterIntent(primary_goal="test"),
            messages=[
                PosterMessage(id="headline", role="headline", content="Title", importance=10, presence="required", source="user"),
            ],
        )
        # Should not raise.
        ContentExtractor()._validate_brief(brief)

    def test_cta_intent_detection(self):
        """_has_cta_intent correctly identifies prompts that need CTA."""
        extractor = ContentExtractor()
        assert extractor._has_cta_intent("立即报名参加活动")
        assert extractor._has_cta_intent("扫码关注我们")
        assert extractor._has_cta_intent("注册成为会员")
        assert not extractor._has_cta_intent("一张安静的艺术海报")
        assert not extractor._has_cta_intent("纯文字排版设计")
        assert not extractor._has_cta_intent("抽象几何图形海报")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0 — StyleDirector baseline
# ═══════════════════════════════════════════════════════════════════════════════


class TestStyleDirectorBaseline:
    """Document the StyleDirector fallback behaviour — Phase 3 poster_type table."""

    def test_tech_prompt_gives_futuristic_mood(self):
        state = GraphState(user_prompt="科技风AI会议海报")
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "futuristic"
        assert guide.primary_color == "#0B1026"

    def test_music_prompt_gives_energetic_mood(self):
        state = GraphState(user_prompt="音乐节演出海报")
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "energetic"
        assert guide.secondary_color == "#FF477E"

    def test_recruitment_prompt_gives_friendly_mood(self):
        state = GraphState(user_prompt="校园招聘海报")
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "friendly"
        assert guide.primary_color == "#F7FAFC"

    def test_unknown_prompt_gives_modern_default(self):
        state = GraphState(user_prompt="一张海报")
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "modern"

    def test_all_colors_are_valid_hex(self):
        """Every colour field must be a valid 6-digit hex."""
        state = GraphState(user_prompt="科技海报")
        guide = StyleDirector()._run_rules(state)
        for field_name in ("primary_color", "secondary_color", "accent_color", "text_color"):
            value = getattr(guide, field_name)
            assert len(value) == 7 and value.startswith("#"), (
                f"{field_name}={value!r} is not a valid hex colour"
            )

    # ── Phase 3: poster_type-based template selection ──

    def test_artistic_poster_type_gives_minimal_style(self):
        """When poster_brief has poster_type='artistic', fallback gives minimal style."""
        state = GraphState(user_prompt="当代艺术展极简海报")
        state.poster_brief = PosterBriefV2(
            poster_intent=PosterIntent(poster_type="artistic", communication_mode="evoke", primary_goal="art exhibition", tone=["minimal"]),
            content_strategy=ContentStrategy(cta_policy="omit"),
        )
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "minimal"
        assert guide.primary_color == "#F5F1EB"  # Off-white background.

    def test_exhibition_poster_type_gives_cultured_style(self):
        """Exhibition posters get a cultured, gallery feel."""
        state = GraphState(user_prompt="当代艺术展海报")
        state.poster_brief = PosterBriefV2(
            poster_intent=PosterIntent(poster_type="exhibition", communication_mode="invite", primary_goal="gallery show"),
            content_strategy=ContentStrategy(cta_policy="omit"),
        )
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "cultured"
        assert guide.font_family == "serif"

    def test_typographic_poster_type_gives_experimental_style(self):
        """Typographic posters get bold, experimental treatment."""
        state = GraphState(user_prompt="纯文字排版海报")
        state.poster_brief = PosterBriefV2(
            poster_intent=PosterIntent(poster_type="typographic", communication_mode="evoke", primary_goal="type poster", tone=["experimental"]),
            content_strategy=ContentStrategy(cta_policy="omit"),
        )
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "experimental"
        # Dark background with light text for high contrast typography.
        assert guide.primary_color == "#0D0D0D"
        assert guide.text_color == "#FAFAFA"

    def test_event_poster_with_music_keyword_gives_energetic(self):
        """Event poster + music keyword → energetic, not generic event template."""
        state = GraphState(user_prompt="音乐节演出海报")
        state.poster_brief = PosterBriefV2(
            poster_intent=PosterIntent(poster_type="event", communication_mode="celebrate", primary_goal="music festival"),
            content_strategy=ContentStrategy(cta_policy="optional"),
        )
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "energetic"
        assert guide.secondary_color == "#FF477E"  # Vibrant pink.

    def test_informational_poster_type_gives_structured_style(self):
        """Informational posters get clear, structured, readable style."""
        state = GraphState(user_prompt="讲座日程信息海报")
        state.poster_brief = PosterBriefV2(
            poster_intent=PosterIntent(poster_type="informational", communication_mode="inform", primary_goal="schedule"),
            content_strategy=ContentStrategy(cta_policy="omit"),
        )
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "professional"
        assert guide.accent_color == "#2563EB"  # Clean blue accent.

    def test_product_poster_type_gives_premium_style(self):
        """Product posters get a clean, premium studio look."""
        state = GraphState(user_prompt="产品发布海报")
        state.poster_brief = PosterBriefV2(
            poster_intent=PosterIntent(poster_type="product", communication_mode="persuade", primary_goal="product launch"),
            content_strategy=ContentStrategy(cta_policy="required"),
        )
        guide = StyleDirector()._run_rules(state)
        assert guide.mood == "premium"
        assert guide.font_family == "sans-serif"

    # ── Phase 3: art_direction conversion ──

    def test_style_guide_to_art_direction_conversion(self):
        """Legacy StyleGuide is converted to ArtDirectionV2 on fallback."""
        state = GraphState(user_prompt="艺术展览")
        state.poster_brief = PosterBriefV2(
            poster_intent=PosterIntent(poster_type="exhibition", communication_mode="invite", primary_goal="exhibition"),
            content_strategy=ContentStrategy(cta_policy="omit"),
        )
        ad = StyleDirector()._style_guide_to_art_direction(
            StyleGuide(
                theme_keywords=["gallery"], background_prompt="museum wall",
                primary_color="#1C1C1C", secondary_color="#8B7E74",
                accent_color="#D4A574", text_color="#F5F0E8",
                font_family="serif", mood="cultured",
            ),
            state,
        )
        assert ad.poster_language.composition_family == "editorial_spread"
        assert ad.color_system.accent == "#D4A574"
        assert ad.typography.headline_style == "grotesk"
        assert ad.imagery.prompt == "museum wall"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0 — SpatialLayoutPlanner baseline
# ═══════════════════════════════════════════════════════════════════════════════


def _plan_with_presence(elements_spec: list[dict]) -> ContentPlan:
    """Build a ContentPlan from a compact spec list."""
    elements = []
    for spec in elements_spec:
        elements.append(ElementContent(
            id=spec["id"],
            type=ElementType(spec.get("type", "text")),
            content=spec.get("content", spec["id"]),
            priority=spec.get("priority", 5),
            role=spec.get("role"),
            presence=spec.get("presence", "required"),
        ))
    return ContentPlan(poster_goal="test", elements=elements)


def _style_for_test() -> StyleGuide:
    return StyleGuide(
        theme_keywords=["test"],
        background_prompt="clean",
        primary_color="#111111",
        secondary_color="#222222",
        accent_color="#FF0000",
        text_color="#FFFFFF",
        mood="modern",
    )


class TestLayoutPlannerBaseline:
    """Document the LayoutPlanner prompt and fallback template selection."""

    def test_prompt_no_longer_forces_all_elements(self):
        """The system prompt must not say 'Every element MUST appear'."""
        state = GraphState(user_prompt="poster", canvas=CanvasSpec(width=512, height=768))
        state.content_plan = _plan_with_presence([
            {"id": "title", "content": "Hello", "priority": 10, "role": "headline", "presence": "required"},
        ])
        state.style = _style_for_test()
        messages = SpatialLayoutPlanner()._build_messages(state)
        system = messages[0]["content"]
        assert "Every element" not in system
        assert "required" in system.lower()
        assert "optional" in system.lower()

    # ── Phase 4: PosterBriefV2 + ArtDirectionV2 integration ──

    def test_prompt_includes_presence_rules_for_each_element(self):
        """User prompt must list element presence rules."""
        state = GraphState(user_prompt="poster", canvas=CanvasSpec(width=512, height=768))
        state.content_plan = _plan_with_presence([
            {"id": "title", "content": "Hello", "priority": 10, "role": "headline", "presence": "required"},
            {"id": "subtitle", "content": "Sub", "priority": 8, "role": "subhead", "presence": "optional"},
        ])
        state.style = _style_for_test()
        messages = SpatialLayoutPlanner()._build_messages(state)
        user = messages[1]["content"]
        assert "ELEMENT PRESENCE RULES" in user
        assert "REQUIRED title" in user
        assert "OPTIONAL subtitle" in user

    def test_prompt_uses_poster_brief_when_available(self):
        """When poster_brief is set, prompt uses it instead of content_plan only."""
        state = GraphState(user_prompt="poster", canvas=CanvasSpec(width=512, height=768))
        state.content_plan = _plan_with_presence([
            {"id": "title", "content": "Hello", "priority": 10, "role": "headline", "presence": "required"},
        ])
        state.style = _style_for_test()
        state.poster_brief = PosterBriefV2(
            poster_intent=PosterIntent(poster_type="artistic", communication_mode="evoke", primary_goal="art poster", tone=["minimal"]),
            content_strategy=ContentStrategy(cta_policy="omit", image_policy="omit"),
            messages=[
                PosterMessage(id="headline", role="headline", content="Art Show", importance=10, presence="required", source="user", editable=False),
            ],
            must_not_do=["不要默认加入报名按钮"],
        )
        messages = SpatialLayoutPlanner()._build_messages(state)
        user = messages[1]["content"]
        assert "PosterBriefV2" in user
        assert "artistic" in user
        assert "MUST NOT" in user

    def test_prompt_uses_art_direction_when_available(self):
        """When art_direction is set, prompt includes composition/colour hints."""
        state = GraphState(user_prompt="poster", canvas=CanvasSpec(width=512, height=768))
        state.content_plan = _plan_with_presence([
            {"id": "title", "content": "Hello", "priority": 10, "role": "headline", "presence": "required"},
        ])
        state.style = _style_for_test()
        state.art_direction = ArtDirectionV2(
            style_name="minimal type",
            mood_keywords=["minimal"],
            poster_language=PosterLanguage(composition_family="typographic", visual_density="sparse", negative_space="generous", depth_strategy="flat", risk_level="expressive"),
            color_system=ColorSystem(background="#0D0D0D", foreground="#FAFAFA", accent="#FF3366", secondary="#E5E5E5", palette_notes="high contrast"),
            typography=TypographySpec(headline_style="grotesk", body_style="sans", scale_contrast="extreme", letter_case="as_given"),
            imagery=ImagerySpec(treatment="none", background_strategy="plain", prompt="", negative_prompt="clutter"),
        )
        messages = SpatialLayoutPlanner()._build_messages(state)
        user = messages[1]["content"]
        assert "ArtDirectionV2" in user
        assert "typographic" in user
        assert "COMPOSITION" in user or "Composition" in user or "composition" in user

    def test_prompt_forbids_landing_page_structure(self):
        """System prompt must explicitly forbid landing-page/card/button defaults."""
        state = GraphState(user_prompt="poster", canvas=CanvasSpec(width=512, height=768))
        state.content_plan = _plan_with_presence([
            {"id": "title", "content": "Hello", "priority": 10, "role": "headline", "presence": "required"},
        ])
        state.style = _style_for_test()
        messages = SpatialLayoutPlanner()._build_messages(state)
        system = messages[0]["content"]
        assert "not a web landing page" in system
        assert "UI card" in system
        assert "OMIT" in system

    def test_fallback_selects_type_only_when_no_cta_no_visual(self):
        """Type-only poster → type_only template (no CTA, no image)."""
        state = GraphState(user_prompt="纯文字海报", canvas=CanvasSpec(width=512, height=768))
        state.content_plan = _plan_with_presence([
            {"id": "title", "content": "Jazz Night", "priority": 10, "role": "headline", "presence": "required"},
        ])
        state.style = _style_for_test()
        html = SpatialLayoutPlanner()._run_fallback(state)
        assert "<!DOCTYPE html>" in html
        assert "Jazz Night" in html
        # Type-only template has no .cta or .visual CSS classes.
        assert ".cta" not in html
        assert ".visual" not in html

    def test_fallback_selects_image_led_when_visual_present_no_cta(self):
        """Image-led poster → image_led template."""
        state = GraphState(user_prompt="视觉海报", canvas=CanvasSpec(width=512, height=768))
        state.content_plan = _plan_with_presence([
            {"id": "title", "content": "Exhibition", "priority": 10, "role": "headline", "presence": "required"},
            {"id": "subtitle", "content": "Modern Art", "priority": 8, "role": "subhead", "presence": "recommended"},
            {"id": "main_visual", "content": "abstract", "priority": 7, "type": "image", "role": "visual_label", "presence": "recommended"},
        ])
        state.style = _style_for_test()
        html = SpatialLayoutPlanner()._run_fallback(state)
        assert "text-zone" in html
        assert "Exhibition" in html
        assert ".cta" not in html  # No CTA in image-led template.

    def test_fallback_selects_event_info_when_cta_no_visual(self):
        """Event poster with CTA but no visual → event_info template."""
        state = GraphState(user_prompt="活动报名", canvas=CanvasSpec(width=512, height=768))
        state.content_plan = _plan_with_presence([
            {"id": "title", "content": "Conference", "priority": 10, "role": "headline", "presence": "required"},
            {"id": "subtitle", "content": "June 2026", "priority": 8, "role": "subhead", "presence": "recommended"},
            {"id": "cta", "content": "Register Now", "priority": 5, "role": "cta", "presence": "required"},
        ])
        state.style = _style_for_test()
        html = SpatialLayoutPlanner()._run_fallback(state)
        assert "left-bar" in html
        assert "Register Now" in html
        assert ".visual" not in html  # No visual in event info template.

    def test_fallback_selects_cta_campaign_when_both_visual_and_cta(self):
        """Full campaign poster → cta_campaign template (original style)."""
        state = GraphState(user_prompt="营销海报", canvas=CanvasSpec(width=512, height=768))
        state.content_plan = _plan_with_presence([
            {"id": "title", "content": "Sale", "priority": 10, "role": "headline", "presence": "required"},
            {"id": "subtitle", "content": "Big Offer", "priority": 8, "role": "subhead", "presence": "recommended"},
            {"id": "main_visual", "content": "product", "priority": 7, "type": "image", "role": "visual_label", "presence": "recommended"},
            {"id": "cta", "content": "Buy Now", "priority": 5, "role": "cta", "presence": "required"},
        ])
        state.style = _style_for_test()
        html = SpatialLayoutPlanner()._run_fallback(state)
        assert "Sale" in html
        assert "Buy Now" in html
        assert ".cta" in html
        assert ".visual" in html


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0 — VLMCritic baseline
# ═══════════════════════════════════════════════════════════════════════════════


class TestVLMCriticBaseline:
    """Document the VLMCritic heuristic fallback behaviour."""

    def test_heuristic_checks_text_elements_in_html(self):
        """Heuristic finds missing text elements and scores accordingly."""
        state = GraphState(user_prompt="poster")
        state.content_plan = ContentPlan(
            poster_goal="test",
            elements=[
                ElementContent(id="title", type=ElementType.text, content="Hello", priority=10),
                ElementContent(id="missing_text", type=ElementType.text, content="Nowhere", priority=5),
            ],
        )
        state.layout_html = "<html><body><h1>Hello</h1></body></html>"
        result = HeuristicVLMCritic()._run_heuristic(state)
        # "Nowhere" text is not in the HTML → should be flagged.
        assert any("missing_text" in issue or "Nowhere" in issue for issue in result.issues)
        assert result.score < 92  # Penalty applied.

    def test_heuristic_checks_html_structure(self):
        """Heuristic flags missing <style> and <body> tags."""
        state = GraphState(user_prompt="poster")
        state.content_plan = ContentPlan(
            poster_goal="test",
            elements=[ElementContent(id="title", type=ElementType.text, content="Title", priority=10)],
        )
        state.layout_html = "just plain text without html structure"
        result = HeuristicVLMCritic()._run_heuristic(state)
        assert any("body" in issue.lower() or "html" in issue.lower() for issue in result.issues)

    def test_heuristic_accepts_well_formed_html(self):
        """Well-formed HTML with all text present scores well."""
        state = GraphState(user_prompt="poster")
        state.content_plan = ContentPlan(
            poster_goal="test",
            elements=[ElementContent(id="title", type=ElementType.text, content="Hello World", priority=10)],
        )
        state.layout_html = "<html><body><style>.t{color:red}</style><h1>Hello World</h1></body></html>"
        result = HeuristicVLMCritic()._run_heuristic(state)
        assert len(result.issues) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0 — ContentExtractor LLM prompt baseline
# ═══════════════════════════════════════════════════════════════════════════════

class FakePromptCaptureClient:
    """Fake LLM client that captures the prompt and returns a valid PosterBriefV2."""
    api_key = "key"
    base_url = "https://example.test/v1"
    model = "real-model"

    def __init__(self):
        self.captured_messages: list = []

    async def parse(self, *, messages, response_model):
        self.captured_messages.append(messages)
        return PosterBriefV2(
            poster_intent=PosterIntent(
                poster_type="artistic", communication_mode="evoke",
                primary_goal="test poster", tone=["minimal"],
            ),
            content_strategy=ContentStrategy(cta_policy="omit"),
            messages=[
                PosterMessage(
                    id="headline", role="headline", content="Title",
                    importance=10, presence="required", source="user",
                ),
            ],
        )


class TestContentExtractorPromptBaseline:
    """Verify the LLM prompt no longer forces marketing-card elements."""

    async def test_llm_prompt_does_not_require_cta(self):
        """System prompt must not hardcode CTA as required."""
        client = FakePromptCaptureClient()
        state = GraphState(user_prompt="艺术展览海报")
        await ContentExtractor(llm_client=client).run(state)
        system = client.captured_messages[0][0]["content"]
        assert "Required ids:" not in system, "Must not force specific element ids"
        assert "Do NOT force" not in system  # Phase 2 prompt wording changed
        assert "Do NOT require" in system
        assert "CTA is required only when" in system
        # Phase 2: poster_brief stored on state.
        assert state.poster_brief is not None

    async def test_llm_prompt_mentions_poster_brief_v2(self):
        """System prompt should instruct LLM to return PosterBriefV2."""
        client = FakePromptCaptureClient()
        state = GraphState(user_prompt="海报")
        await ContentExtractor(llm_client=client).run(state)
        system = client.captured_messages[0][0]["content"]
        assert "PosterBriefV2" in system
        assert "poster_intent" in system
        assert "content_strategy" in system
        assert "cta_policy" in system
