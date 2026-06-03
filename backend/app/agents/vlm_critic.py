"""Vision-language critic — Phase 5 (poster-specific rubric).

Critiques a rendered poster by sending the PNG image to a vision model along
with the PosterBriefV2, ArtDirectionV2, and HTML layout context.  Uses
``parse_vision`` with *enable_thinking* so the model can reason before scoring.

The critique now includes a dimension-level rubric, structured issues, and a
``revision_focus`` field that directly drives the router.
"""

from __future__ import annotations

import json
import re

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.llm_client import StructuredLLMClient
from app.schemas.agents import CritiqueIssue, CritiqueResult
from app.schemas.state import GraphState


class HeuristicVLMCritic:
    """Critique a rendered poster using a vision-language model.

    Sends the rendered PNG along with poster_brief, art_direction, and HTML
    layout snippet.  With *enable_thinking* the model first reasons about what
    it sees (stored in ``state.vision_reasoning``) and then emits a
    ``CritiqueResult`` with rubric, structured issues, and revision_focus.

    Falls back to a deterministic heuristic when the vision model is
    unavailable.
    """

    def __init__(self, vision_client: StructuredLLMClient | None = None) -> None:
        settings = get_settings()
        self.allow_model_fallback = settings.allow_model_fallback
        self.enable_thinking = settings.vision_enable_thinking
        self.thinking_budget = settings.vision_thinking_budget
        self.html_context_chars = settings.vision_html_context_chars
        self._vision_model_warning = _vision_model_warning(settings.vision_model)
        self.vision_client = vision_client or StructuredLLMClient(
            api_key=settings.vision_api_key,
            base_url=settings.vision_base_url,
            model=settings.vision_model,
            timeout=settings.vision_timeout_seconds,
            response_format=settings.llm_response_format,
            max_tokens=settings.vision_max_tokens,
        )

    # ── public entry point ──

    async def run(self, state: GraphState) -> CritiqueResult:
        if self._configured_for_vision() and self._vision_model_warning:
            warning = self._vision_model_warning
            if warning not in state.warnings:
                state.warnings.append(warning)
        try:
            return await self._run_vision_model(state)
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_vision() and not self.allow_model_fallback:
                raise
            if self._configured_for_vision():
                state.warnings.append(f"VLMCritic vision fallback: {exc}")
            return self._run_heuristic(state)

    # ── vision model path (Phase 5 V2 prompt) ──

    async def _run_vision_model(self, state: GraphState) -> CritiqueResult:
        if state.render_result is None or not state.render_result.image_base64:
            raise LLMCallError("render result image_base64 is required for vision critique")
        if state.layout_html is None or state.content_plan is None:
            raise LLMCallError("layout_html and content_plan are required for vision critique")

        image_url = f"data:{state.render_result.mime_type};base64,{state.render_result.image_base64}"
        messages = self._build_vision_messages(state, image_url)

        result, reasoning = await self.vision_client.parse_vision(
            messages=messages,
            response_model=CritiqueResult,
            enable_thinking=self.enable_thinking,
            thinking_budget=self.thinking_budget,
        )

        if reasoning:
            state.vision_reasoning = reasoning

        return result

    def _build_vision_messages(self, state: GraphState, image_url: str) -> list[dict]:
        """Construct the multi-modal messages — Phase 5 with brief + art direction."""
        html_snippet = state.layout_html[:self.html_context_chars] if state.layout_html else ""

        brief_json = state.poster_brief.model_dump(mode="json") if state.poster_brief else None
        ad_json = state.art_direction.model_dump(mode="json") if state.art_direction else None
        visual_system_json = (
            state.visual_system.model_dump(mode="json") if state.visual_system else None
        )

        system_prompt = (
            "You are a strict poster art director reviewing a rendered poster image.\n"
            "Return only JSON matching CritiqueResult.\n\n"
            "Evaluate it as a POSTER, not as a web page.\n\n"
            "Step 1 — Describe what you literally see in the image: composition, "
            "text content, imagery, colour, hierarchy, spacing, and style. "
            "Put this in the **vision_description** field.\n\n"
            "Step 2 — Judge whether the poster type and communication mode match "
            "the PosterBriefV2.\n\n"
            "Step 3 — Score the poster (0-100) using the **rubric** with these "
            "dimensions (each 0-20, total normalised):\n"
            "  - poster_identity: does it feel like a finished poster?\n"
            "  - topic_fit: does it express the user's theme?\n"
            "  - composition: is there a clear visual idea and hierarchy?\n"
            "  - typography: is type intentional and suitable?\n"
            "  - readability: can required information be read?\n"
            "  - craft: does it avoid broken rendering, overlap, and generic template feel?\n"
            "Explain your scoring in the **reasoning** field. "
            "Set **passed**=true if the poster is good enough to ship.\n\n"
            "Be strict about visual richness. Unless PosterBriefV2 asks for "
            "extreme minimalism, penalize posters that look like sparse templates: "
            "plain gradient background, centered text only, one small generic icon, "
            "empty placeholder visual, weak hierarchy, or no texture/layering. "
            "A finished poster should have a clear compositional idea and enough "
            "visual craft to feel intentional.\n\n"
            "If a VisualSystemPlan is provided, check whether the required layer "
            "ids and the planned layer count appear to be implemented. Penalize "
            "posters that ignore the plan, merge everything into a single vague "
            "background, or omit required_html_ids.\n\n"
            "The top-level **score** field MUST be a single integer from 0 to 100. "
            "Put the dimension scores in the separate **rubric** object. "
            "Never put the rubric object inside the score field.\n\n"
            "Step 4 — List concrete visible issues as **structured_issues**. Each "
            "must have: type (composition|typography|content|color|imagery|rendering|style), "
            "severity (minor|major|blocking), target_id (element id, or null), "
            "description, and suggestion.\n"
            "Also populate the legacy **issues** (string list) and **suggestions** "
            "(string list) with the same content.\n\n"
            "Step 5 — Set **revision_focus** to one of:\n"
            '  - "final" — poster is good enough, stop iterating\n'
            '  - "layout" — needs layout/composition/spacing/topology fixes\n'
            '  - "style" — needs colour, mood, background, or font changes\n'
            '  - "content" — missing required text, wrong information, or content mismatch\n'
            '  - "render" — has browser rendering errors, broken fonts, or CSS bugs\n\n'
            "IMPORTANT RULES:\n"
            "- Do NOT demand a CTA, subtitle, hero image, or button unless the "
            "PosterBriefV2 marks them as required (presence='required').\n"
            "- Do NOT penalize intentional minimalism, type-only design, or abstract "
            "composition when it matches the brief.\n"
            "- Do NOT invent issues that are not visible in the image.\n"
            "- If the poster is genuinely well-done and matches the brief, give it "
            "a high score and set revision_focus='final'.\n\n"
            "Return ONLY a JSON object matching CritiqueResult — no markdown, no extras.\n"
            "Expected shape example:\n"
            '{"score": 88, "passed": true, "reasoning": "...", '
            '"vision_description": "...", "issues": [], "suggestions": [], '
            '"structured_issues": [], '
            '"rubric": {"poster_identity": 18, "topic_fit": 18, '
            '"composition": 17, "typography": 17, "readability": 18, "craft": 18}, '
            '"revision_focus": "final", "do_not_change": []}'
        )

        user_text = (
            f"Poster brief (PosterBriefV2):\n{json.dumps(brief_json, ensure_ascii=False) if brief_json else 'none'}\n\n"
            f"Art direction (ArtDirectionV2):\n{json.dumps(ad_json, ensure_ascii=False) if ad_json else 'none'}\n\n"
            f"Visual system plan (VisualSystemPlan):\n{json.dumps(visual_system_json, ensure_ascii=False) if visual_system_json else 'none'}\n\n"
            f"HTML layout (first {self.html_context_chars} chars):\n{html_snippet}\n\n"
            "Review the rendered poster image and return a complete CritiqueResult."
        )

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

    # ── heuristic fallback (Phase 5 — respects cta_policy) ──

    def _run_heuristic(self, state: GraphState) -> CritiqueResult:
        """Deterministic critique when the vision model is not available.

        Checks the HTML against content_plan and poster_brief.
        Does NOT penalize missing CTA unless cta_policy is 'required'.
        """
        if not state.content_plan or not state.content_plan.elements:
            raise ValueError("content_plan with elements is required before critique")

        elements = state.content_plan.elements
        html = state.layout_html or ""
        legacy_issues: list[str] = []
        legacy_suggestions: list[str] = []
        structured: list[CritiqueIssue] = []

        brief = state.poster_brief
        cta_required = (
            brief.content_strategy.cta_policy == "required" if brief else False
        )

        # ── Check text elements ──
        for el in elements:
            if el.type.value == "text" and el.content:
                if el.content not in html:
                    # Skip CTA check if CTA is not required.
                    if (el.role == "cta" or el.id == "cta") and not cta_required:
                        continue
                    # Only flag required/recommended elements as issues.
                    if el.presence in ("required", "recommended"):
                        desc = f"Text element '{el.id}' ({el.content}) may be missing from HTML"
                        legacy_issues.append(desc)
                        legacy_suggestions.append(
                            f"Ensure the text '{el.content}' appears in the HTML body for element '{el.id}'."
                        )
                        structured.append(CritiqueIssue(
                            type="content", severity="major", target_id=el.id,
                            description=desc,
                            suggestion=f"Add '{el.content}' to the HTML body.",
                        ))

        # ── Check HTML structure ──
        if "<body" not in html.lower() and "<html" not in html.lower():
            legacy_issues.append("HTML is missing <body> or <html> tags")
            legacy_suggestions.append("Add proper <!DOCTYPE html> and <body> structure to the HTML.")
            structured.append(CritiqueIssue(
                type="rendering", severity="blocking", target_id=None,
                description="HTML is missing <body> or <html> tags",
                suggestion="Add proper <!DOCTYPE html> and <body> structure.",
            ))

        if "<style" not in html and "style=" not in html:
            legacy_issues.append("No CSS styling found in the HTML")
            legacy_suggestions.append("Add inline CSS or a <style> block for visual design.")
            structured.append(CritiqueIssue(
                type="style", severity="major", target_id=None,
                description="No CSS styling found in the HTML",
                suggestion="Add inline CSS or a <style> block for visual design.",
            ))

        # Real generation states carry PosterBriefV2/ArtDirectionV2. In that
        # path, also reject sparse placeholder posters that pass basic HTML checks.
        if (brief is not None or state.art_direction is not None) and not self._allows_extreme_minimalism(state):
            layer_count = self._visual_layer_count(html)
            has_placeholder_icon = self._looks_like_placeholder_icon(html)
            if layer_count < 4 or has_placeholder_icon:
                desc = (
                    "Poster appears visually underdeveloped: it has too few distinct "
                    "layers or relies on a generic placeholder visual."
                )
                legacy_issues.append(desc)
                legacy_suggestions.append(
                    "Add a stronger poster system: cropped type or symbol, texture/grid, "
                    "directional shapes, richer foreground/background layering, and a clearer focal idea."
                )
                structured.append(CritiqueIssue(
                    type="style",
                    severity="major",
                    target_id=None,
                    description=desc,
                    suggestion=(
                        "Increase visual richness with at least four purposeful layers "
                        "and replace generic placeholder icons with a scaled, cropped, "
                        "theme-specific visual device."
                    ),
                ))

        if state.visual_system is not None:
            missing_layer_ids = [
                layer_id
                for layer_id in state.visual_system.required_html_ids
                if layer_id and not self._html_mentions_identifier(html, layer_id)
            ]
            if missing_layer_ids:
                desc = (
                    "HTML appears to ignore required VisualSystemPlan layers: "
                    + ", ".join(missing_layer_ids[:6])
                )
                legacy_issues.append(desc)
                legacy_suggestions.append(
                    "Implement the required visual-system layers with matching id or data-layer-id attributes."
                )
                structured.append(CritiqueIssue(
                    type="composition",
                    severity="major",
                    target_id=None,
                    description=desc,
                    suggestion=(
                        "Add visible HTML/CSS layers for the missing VisualSystemPlan ids, "
                        "rather than merging them into a single background."
                    ),
                ))

            if not self._allows_extreme_minimalism(state):
                layer_count = self._visual_layer_count(html)
                target = max(4, min(state.visual_system.layer_count_target - 1, 7))
                if layer_count < target:
                    desc = (
                        f"Visual layer signal is too low for the planned system "
                        f"({layer_count} cues found, target about {target})."
                    )
                    legacy_issues.append(desc)
                    legacy_suggestions.append(
                        "Use the VisualSystemPlan to add distinct texture, focal, shape, frame, and metadata layers."
                    )
                    structured.append(CritiqueIssue(
                        type="style",
                        severity="major",
                        target_id=None,
                        description=desc,
                        suggestion=(
                            "Increase distinct visual layers according to the VisualSystemPlan "
                            "instead of relying on a sparse centered composition."
                        ),
                    ))

        # ── Determine revision_focus ──
        if not structured:
            revision_focus = "final"
        elif any(i.type == "rendering" or i.severity == "blocking" for i in structured):
            revision_focus = "render"
        elif any(i.type == "content" for i in structured):
            revision_focus = "content"
        elif any(i.type == "style" for i in structured):
            revision_focus = "style"
        else:
            revision_focus = "layout"

        score = max(60, 92 - len(structured) * 8)
        passed = score >= state.target_score
        reasoning = (
            "Layout contains expected elements and styling."
            if passed
            else "Layout is missing key elements or styling."
        )

        return CritiqueResult(
            score=score,
            passed=passed,
            reasoning=reasoning,
            vision_description="(heuristic — no vision model available)",
            issues=legacy_issues,
            suggestions=legacy_suggestions,
            structured_issues=structured,
            revision_focus=revision_focus,
        )

    # ── helpers ──

    def _configured_for_vision(self) -> bool:
        return bool(
            self.vision_client.api_key
            and self.vision_client.base_url
            and not self.vision_client.model.startswith("mock-")
        )

    def _allows_extreme_minimalism(self, state: GraphState) -> bool:
        """Return True when a sparse poster is likely intentional."""
        brief = state.poster_brief
        ad = state.art_direction
        tone = [item.lower() for item in (brief.poster_intent.tone if brief else [])]
        poster_type = (brief.poster_intent.poster_type if brief else "").lower()
        prompt = state.user_prompt.lower()

        if any(token in prompt for token in ("极简", "minimal", "纯文字", "type-only", "typographic")):
            return True
        if poster_type in {"typographic", "artistic"} and any(token in tone for token in ("minimal", "sparse")):
            return True
        if ad is not None:
            pl = ad.poster_language
            if (
                pl.visual_density == "sparse"
                and pl.negative_space == "generous"
                and pl.composition_family in {"minimal", "typographic"}
            ):
                return True
        return False

    def _visual_layer_count(self, html: str) -> int:
        """Rough CSS/HTML signal for whether the poster has visual craft."""
        lower = html.lower()
        cues = (
            "radial-gradient",
            "repeating-linear-gradient",
            "background-image",
            "mix-blend-mode",
            "clip-path",
            "box-shadow",
            "filter:",
            "transform:",
            "<svg",
            "<img",
            "::before",
            "::after",
            "border:",
            "grid-template",
            "texture",
            "grain",
            "pattern",
            "frame",
            "rule",
            "metadata",
            "data-row",
        )
        return sum(1 for cue in cues if cue in lower)

    def _html_mentions_identifier(self, html: str, identifier: str) -> bool:
        """Return True when a visual-system id appears as id, data-layer-id, or class."""
        lower = html.lower()
        ident = re.escape(identifier.lower())
        patterns = (
            rf'id\s*=\s*["\']{ident}["\']',
            rf'data-layer-id\s*=\s*["\']{ident}["\']',
            rf'class\s*=\s*["\'][^"\']*\b{ident}\b[^"\']*["\']',
        )
        return any(re.search(pattern, lower) for pattern in patterns)

    def _looks_like_placeholder_icon(self, html: str) -> bool:
        """Detect the old fallback style: a single faint circle as key visual."""
        lower = html.lower()
        circle_count = lower.count("<circle")
        path_count = lower.count("<path")
        has_visual = 'data-role="visual"' in lower or "class=\"visual\"" in lower
        small_svg = bool(re.search(r"\.visual\s+svg\s*\{[^}]*width:\s*3\d%", lower))
        return has_visual and circle_count <= 2 and path_count == 0 and small_svg


def _vision_model_warning(model: str) -> str:
    model_lower = model.lower()
    if model_lower.startswith("mock-"):
        return ""
    vision_tokens = ("vl", "vision", "omni")
    if any(token in model_lower for token in vision_tokens):
        return ""
    return (
        f"VISION_MODEL={model} does not look like a vision-capable model. "
        "Use a VL/vision model such as qwen3-vl-flash, qwen3-vl-plus, or qwen-vl-max-latest."
    )
