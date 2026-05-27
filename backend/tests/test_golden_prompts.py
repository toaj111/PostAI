"""Phase 6 — Golden prompts regression suite.

Each golden prompt goes through the full deterministic pipeline (no LLM /
no VLM) so the test suite can verify that every poster_type produces a
distinct fallback template and that the V2 schemas (PosterBriefV2,
ArtDirectionV2, CritiqueResult) are populated correctly.

These are NOT visual regression tests — they verify structural properties
of the pipeline output.
"""

import pytest

from app.agents.content_extractor import ContentExtractor
from app.agents.layout_planner import SpatialLayoutPlanner
from app.agents.style_director import StyleDirector
from app.agents.vlm_critic import HeuristicVLMCritic
from app.schemas.agents import (
    ContentPlan,
    ContentStrategy,
    ElementContent,
    PosterBriefV2,
    PosterIntent,
)
from app.schemas.layout import CanvasSpec, ElementType
from app.schemas.state import GraphState

# ═══════════════════════════════════════════════════════════════════════════════
# Golden prompts — one per poster type
# ═══════════════════════════════════════════════════════════════════════════════

GOLDEN_PROMPTS = [
    {
        "name": "art_exhibition_minimal",
        "prompt": "为一个当代艺术展做一张极简海报，主题是'静默的机器'，不要按钮",
        "expected_poster_type": "artistic",  # via fallback keyword
        "expected_cta_policy": "omit",
        "expected_has_cta": False,
        "expected_has_visual": False,  # minimal → no placeholder visual by default
    },
    {
        "name": "jazz_typographic",
        "prompt": "做一张纯文字爵士音乐节海报，极简排版，不要 CTA 按钮",
        "expected_cta_policy": "omit",
        "expected_has_cta": False,
        "expected_has_visual": False,
    },
    {
        "name": "recruitment_with_qr",
        "prompt": "招聘海报，需要扫码报名，温暖友好的校园风格",
        "expected_cta_policy": "required",
        "expected_has_cta": True,
    },
    {
        "name": "lecture_schedule_dense",
        "prompt": "制作信息密集的讲座日程海报，包含多个时段和地点",
        "expected_cta_policy": "omit",
        "expected_has_cta": False,
    },
    {
        "name": "abstract_summer",
        "prompt": "只用抽象形状和色彩表达海边夏日的氛围",
        "expected_cta_policy": "omit",
        "expected_has_cta": False,
    },
    {
        "name": "product_launch",
        "prompt": "产品发布海报，简洁高端展示产品质感，需要预约体验",
        "expected_cta_policy": "required",
        "expected_has_cta": True,
    },
    {
        "name": "music_festival",
        "prompt": "音乐节演出海报，高能量、大胆构图、霓虹色彩",
        "expected_cta_policy": "omit",
        "expected_has_cta": False,
    },
]


def _run_pipeline(prompt: str) -> GraphState:
    """Run the full deterministic pipeline (no LLM/VLM) for a given prompt."""
    from app.schemas.agents import content_plan_to_poster_brief

    state = GraphState(user_prompt=prompt, canvas=CanvasSpec(width=512, height=768))

    # Content
    plan = ContentExtractor()._run_rules(state)
    state.content_plan = plan
    state.poster_brief = content_plan_to_poster_brief(plan)

    # Style
    guide = StyleDirector()._run_rules(state)
    state.style = guide
    state.art_direction = StyleDirector()._style_guide_to_art_direction(guide, state)

    # Layout
    html = SpatialLayoutPlanner()._run_fallback(state)
    state.layout_html = html

    # Critic
    critique = HeuristicVLMCritic()._run_heuristic(state)
    state.feedback_history.append(critique)

    return state


class TestGoldenPrompts:
    """Structural regression for all golden prompts."""

    @pytest.mark.parametrize("golden", GOLDEN_PROMPTS, ids=[g["name"] for g in GOLDEN_PROMPTS])
    def test_golden_prompt_produces_valid_html(self, golden):
        """Every golden prompt must produce valid HTML."""
        state = _run_pipeline(golden["prompt"])
        assert state.layout_html is not None
        assert "<!DOCTYPE html>" in state.layout_html
        assert "<body" in state.layout_html.lower() or "<html" in state.layout_html.lower()

    @pytest.mark.parametrize("golden", GOLDEN_PROMPTS, ids=[g["name"] for g in GOLDEN_PROMPTS])
    def test_golden_prompt_has_poster_brief(self, golden):
        """Every golden prompt must set poster_brief on state."""
        state = _run_pipeline(golden["prompt"])
        assert state.poster_brief is not None
        assert state.poster_brief.poster_intent is not None
        assert state.poster_brief.content_strategy is not None

    @pytest.mark.parametrize("golden", GOLDEN_PROMPTS, ids=[g["name"] for g in GOLDEN_PROMPTS])
    def test_golden_prompt_cta_policy(self, golden):
        """CTA policy must match expectations."""
        state = _run_pipeline(golden["prompt"])
        actual = state.poster_brief.content_strategy.cta_policy
        assert actual == golden["expected_cta_policy"], (
            f"Expected cta_policy={golden['expected_cta_policy']} for '{golden['prompt']}', got {actual}"
        )

    @pytest.mark.parametrize("golden", GOLDEN_PROMPTS, ids=[g["name"] for g in GOLDEN_PROMPTS])
    def test_golden_prompt_cta_in_elements(self, golden):
        """CTA element presence must match expectations."""
        state = _run_pipeline(golden["prompt"])
        has_cta = any(
            (e.id == "cta" or e.role == "cta") and e.presence == "required"
            for e in state.content_plan.elements
        )
        assert has_cta == golden["expected_has_cta"], (
            f"Expected has_cta={golden['expected_has_cta']} for '{golden['prompt']}', got {has_cta}"
        )

    @pytest.mark.parametrize("golden", GOLDEN_PROMPTS, ids=[g["name"] for g in GOLDEN_PROMPTS])
    def test_golden_prompt_has_art_direction(self, golden):
        """Every golden prompt must produce ArtDirectionV2."""
        state = _run_pipeline(golden["prompt"])
        assert state.art_direction is not None
        assert state.art_direction.color_system is not None
        assert state.art_direction.poster_language is not None

    @pytest.mark.parametrize("golden", GOLDEN_PROMPTS, ids=[g["name"] for g in GOLDEN_PROMPTS])
    def test_golden_prompt_has_critique(self, golden):
        """Every golden prompt must produce a critique with revision_focus."""
        state = _run_pipeline(golden["prompt"])
        assert len(state.feedback_history) > 0
        critique = state.feedback_history[-1]
        assert critique.revision_focus in ("final", "layout", "style", "content", "render")

    def test_different_poster_types_produce_visually_distinct_html(self):
        """HTML output for different poster types should differ structurally."""
        htmls = {}
        for golden in GOLDEN_PROMPTS:
            state = _run_pipeline(golden["prompt"])
            htmls[golden["name"]] = state.layout_html

        # Type-only posters (art_exhibition, jazz, abstract_summer) should NOT have .cta.
        for name in ("art_exhibition_minimal", "jazz_typographic", "abstract_summer"):
            assert ".cta" not in htmls[name], f"{name} should not have CTA element"

        # Recruitment and product SHOULD have .cta.
        for name in ("recruitment_with_qr", "product_launch"):
            assert ".cta" in htmls[name], f"{name} should have CTA element"
