"""Spatial layout planner — Phase 4 (composition-aware HTML/CSS).

Produces a complete, self-contained HTML document that the ``HTMLPainter``
renders via headless Chromium.  The LLM now receives the full
``PosterBriefV2`` and ``ArtDirectionV2`` so it can work as a poster designer
rather than an element-placer.

Every iteration calls the LLM with full context (VLM feedback from the
previous round) so the design improves incrementally.
"""

from __future__ import annotations

import json
import re

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.llm_client import StructuredLLMClient
from app.render.html_painter import _build_fallback_html
from app.schemas.state import GraphState


class SpatialLayoutPlanner:
    """Plan the visual layout of a poster as a self-contained HTML document.

    On every call to ``run()`` the planner asks the text LLM to produce a
    complete HTML page.  VLM feedback from a previous iteration is included
    in the prompt so the LLM knows what to improve.

    Falls back to a template-built HTML document when the LLM is unavailable.
    """

    def __init__(self, llm_client: StructuredLLMClient | None = None) -> None:
        settings = get_settings()
        self.allow_model_fallback = settings.allow_model_fallback
        self.llm_client = llm_client or StructuredLLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            response_format=settings.llm_response_format,
        )

    # ── public API ──

    async def run(self, state: GraphState) -> str:
        """Return a complete HTML document for the poster.

        Always calls the LLM so it can react to the latest VLM feedback.
        """
        try:
            return await self._run_llm(state)
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_llm() and not self.allow_model_fallback:
                raise
            if self._configured_for_llm():
                state.warnings.append(f"SpatialLayoutPlanner LLM fallback: {exc}")
            return self._run_fallback(state)

    # ── LLM path (Phase 4 — PosterBriefV2 + ArtDirectionV2) ──

    async def _run_llm(self, state: GraphState) -> str:
        if state.content_plan is None or state.style is None:
            raise ValueError("content_plan and style are required before layout planning")
        if not self._configured_for_llm():
            raise LLMCallError("LLM provider is not configured")

        messages = self._build_messages(state)
        content = await self.llm_client._chat_completion(
            messages=messages,
            response_model=type("_Dummy", (), {}),
            force_raw=True,
        )
        html = self._extract_html(content)
        self._validate_html(html)
        return html

    def _build_messages(self, state: GraphState) -> list[dict]:
        """Build the LLM prompt — Phase 4 with PosterBriefV2 + ArtDirectionV2."""
        w = state.canvas.width
        h = state.canvas.height
        style = state.style
        ad = state.art_direction
        brief = state.poster_brief

        # ── Colours: prefer ArtDirectionV2, fall back to StyleGuide ──
        if ad is not None:
            cs = ad.color_system
            bg_color = cs.background
            fg_color = cs.foreground
            accent_color = cs.accent
            secondary_color = cs.secondary
            color_notes = cs.palette_notes
        else:
            bg_color = style.primary_color
            fg_color = style.text_color
            accent_color = style.accent_color
            secondary_color = style.secondary_color
            color_notes = ""

        # ── Composition & style hints from ArtDirectionV2 ──
        comp_hints = ""
        if ad is not None:
            pl = ad.poster_language
            comp_hints = (
                f"Composition: {pl.composition_family}, density={pl.visual_density}, "
                f"negative_space={pl.negative_space}, depth={pl.depth_strategy}, "
                f"risk_level={pl.risk_level}.\n"
                f"Typography: headline={ad.typography.headline_style}, "
                f"body={ad.typography.body_style}, "
                f"scale_contrast={ad.typography.scale_contrast}, "
                f"letter_case={ad.typography.letter_case}.\n"
                f"Imagery: treatment={ad.imagery.treatment}, "
                f"background={ad.imagery.background_strategy}.\n"
            )

        # ── Omission rules from PosterBriefV2 (or ContentPlan fallback) ──
        omission_lines: list[str] = []
        if brief is not None:
            for msg in brief.messages:
                if msg.presence == "omit":
                    omission_lines.append(f"- OMIT {msg.id} ({msg.role}): {msg.content}. Reason: {msg.notes or 'not needed for this poster type'}")
                elif msg.presence == "optional":
                    omission_lines.append(f"- OPTIONAL {msg.id} ({msg.role}, importance={msg.importance}): {msg.content}. May omit if composition benefits.")
                elif msg.presence == "required":
                    omission_lines.append(f"- REQUIRED {msg.id} ({msg.role}, importance={msg.importance}): {msg.content}.")
            for vs in brief.visual_subjects:
                line = f"- {vs.presence.upper()} visual {vs.id} ({vs.role}): {vs.description}"
                if vs.avoid:
                    line += f". Avoid: {', '.join(vs.avoid)}"
                omission_lines.append(line)
            for rule in brief.must_not_do:
                omission_lines.append(f"- MUST NOT: {rule}")
        elif state.content_plan is not None:
            # Fallback: derive presence rules from ContentPlan elements.
            for el in state.content_plan.elements:
                label = el.presence.upper()
                omission_lines.append(
                    f"- {label} {el.id} (role={el.role or 'none'}, priority={el.priority}): {el.content}."
                )

        omission_text = "\n".join(omission_lines) if omission_lines else "(no presence rules — use your best judgement)"

        # ── System prompt (Phase 4 V2) ──
        system = (
            "You are a senior poster designer and HTML/CSS production artist.\n"
            "Output one complete self-contained HTML document, starting with "
            "<!DOCTYPE html>.\n"
            "Do NOT wrap the answer in markdown. Do NOT use JavaScript.\n\n"
            f"The body IS the poster canvas: exactly {w}px by {h}px with "
            "overflow hidden.\n\n"
            "DESIGN this poster according to the PosterBriefV2 and ArtDirectionV2 "
            "provided below. First decide the composition archetype internally "
            "(swiss grid, diagonal energy, editorial spread, typographic, "
            "cinematic, brutalist, minimal, centered iconic, collage, etc.), "
            "then implement it in HTML/CSS.\n\n"
            "CRITICAL RULES:\n"
            "- Make a real POSTER, not a web landing page, dashboard, slide, "
            "UI card, or documentation example.\n"
            "- Do NOT force a CTA button, card, hero image, subtitle, or rounded "
            "box unless the element's presence is REQUIRED or the composition "
            "genuinely demands it.\n"
            "- REQUIRED elements → MUST appear.\n"
            "- RECOMMENDED elements → should appear if composition benefits.\n"
            "- OPTIONAL elements → may be omitted when they weaken the design.\n"
            "- OMIT elements → MUST NOT appear.\n"
            "- Preserve the visual hierarchy from importance/priority values.\n\n"
            "CSS GUIDANCE:\n"
            f"- Set html and body to exactly width:{w}px; height:{h}px; "
            "margin:0; padding:0; overflow:hidden.\n"
            "- Do NOT add responsive @media rules, viewport scaling, zoom, "
            "or transform: scale(...) on html/body/the poster root. This is "
            "a fixed-size print poster, not a responsive web page.\n"
            "- Use CSS creatively: typography scale, alignment, grids, masks, "
            "blend modes, texture, borders, cropping, layered composition.\n"
            "- Gradients and shadows are tools, not defaults — use them only "
            "when they serve the composition.\n"
            "- Avoid generic UI cards and rounded button pills unless the "
            "poster concept explicitly calls for them.\n"
            "- Prefer robust system font stacks for Chinese text. "
            "Google Fonts @import is optional.\n"
            "- For images, use inline SVG or placeholder images "
            "(e.g. https://placehold.co/600x400/EEE/31343C).\n"
            "- Ensure text is readable unless the brief intentionally asks "
            "for experimental illegibility.\n\n"
            f"COLOUR PALETTE:\n"
            f"  background={bg_color}, foreground={fg_color}, "
            f"accent={accent_color}, secondary={secondary_color}"
            + (f"\n  notes: {color_notes}" if color_notes else "") + "\n\n"
            f"{comp_hints}\n"
            "IMAGES:\n"
            "- Use reference image URLs only when the brief/art direction calls for them.\n"
            "- Use object-fit and intentional cropping. Keep each image within "
            "15%-35% of canvas area, maintain safe margins, avoid covering "
            "required text.\n"
            "- If no image is needed, make the poster work through type, shape, "
            "colour, or texture.\n\n"
            "STABLE ELEMENT IDs (required for future refinement):\n"
            "- Every major element MUST carry a unique id attribute that matches "
            "its semantic role, e.g.:\n"
            '  <div id="headline" data-role="headline">...</div>\n'
            '  <div id="subtitle" data-role="subhead">...</div>\n'
            '  <div id="key-visual" data-role="visual">...</div>\n'
            '  <div id="cta" data-role="cta">...</div>\n'
            "- Use these exact id values for the corresponding elements. "
            "Do NOT use auto-generated or random IDs.\n"
            "- Add data-role attributes matching the element's role from the "
            "PosterBriefV2.\n\n"
            "IMPORTANT: Return ONLY the HTML source, starting with <!DOCTYPE html>. "
            "Do NOT wrap it in markdown code fences."
        )

        # ── User prompt — structured context ──
        brief_json = brief.model_dump(mode="json") if brief else None
        ad_json = ad.model_dump(mode="json") if ad else None

        user_parts = [
            f"User prompt: {state.user_prompt}",
            f"Canvas: {w}x{h}px",
            "",
            f"Poster brief (PosterBriefV2):\n{json.dumps(brief_json, ensure_ascii=False) if brief_json else 'none'}",
            "",
            f"Art direction (ArtDirectionV2):\n{json.dumps(ad_json, ensure_ascii=False) if ad_json else 'none'}",
            "",
            "ELEMENT PRESENCE RULES (mandatory):",
            omission_text,
        ]

        # ── Feedback context from VLM critique ──
        feedback_text = self._build_feedback_context(state)
        if feedback_text:
            user_parts.append("")
            user_parts.append(feedback_text)

        # ── Reference images ──
        if state.reference_images:
            refs = [
                f"{index}. {image.url} | {image.description}"
                for index, image in enumerate(state.reference_images, start=1)
            ]
            user_parts.append("")
            user_parts.append(
                "Reference images (available for <img> or CSS background-image "
                "if the brief calls for them):\n" + "\n".join(refs)
            )

        user_parts.append("")
        user_parts.append("Create the complete revised HTML poster now. "
                          "Remember: this is a POSTER, not a landing page.")

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def _build_feedback_context(self, state: GraphState) -> str:
        """Build structured VLM feedback context for the LLM prompt."""
        if not state.feedback_history:
            return ""

        latest_fb = state.feedback_history[-1]
        parts: list[str] = []

        if state.render_result and state.render_result.console_errors:
            parts.append(
                "Browser console errors from your previous HTML: "
                + json.dumps(state.render_result.console_errors, ensure_ascii=False)
            )
        if state.vision_reasoning:
            parts.append(f"Vision model reasoning: {state.vision_reasoning[:2000]}")
        if latest_fb.vision_description:
            parts.append(f"What the vision model saw: {latest_fb.vision_description}")
        if latest_fb.issues:
            parts.append(f"Issues found: {json.dumps(latest_fb.issues, ensure_ascii=False)}")
        if latest_fb.suggestions:
            parts.append(
                f"Suggested improvements: {json.dumps(latest_fb.suggestions, ensure_ascii=False)}"
            )

        if not parts:
            return ""

        parts.insert(0, "=== FEEDBACK from VLM review of the PREVIOUS version ===")
        parts.append(
            "Rewrite the HTML to address these specific issues. "
            "Keep the overall structure and content, but fix the problems "
            "described above. Output the complete revised HTML document."
        )
        return "\n".join(parts)

    def _extract_html(self, content: str) -> str:
        """Strip markdown code fences if the LLM wraps the HTML."""
        html = content.strip()
        html = re.sub(r"^```html?\s*\n?", "", html, flags=re.IGNORECASE)
        html = re.sub(r"\n?```\s*$", "", html)
        html = html.strip()
        if not html:
            raise SchemaParseError("LLM returned empty HTML after stripping fences")
        return html

    def _validate_html(self, html: str) -> None:
        """Guard against obviously broken HTML before handing it to the Painter."""
        lower = html.lower()
        if "<body" not in lower and "<html" not in lower:
            raise SchemaParseError("HTML output must contain a <body> or <html> tag")
        if len(html) < 50:
            raise SchemaParseError("HTML output is too short to be a valid poster document")

    # ── fallback (Phase 4 — reads poster_brief for presence rules) ──

    def _run_fallback(self, state: GraphState) -> str:
        """Template-built poster when the LLM is not available.

        Reads from content_plan elements (legacy) and poster_brief (Phase 2+)
        to select the right template.
        """
        style = state.style
        elements = state.content_plan.elements if state.content_plan else []
        brief = state.poster_brief

        title = "Poster"
        subtitle = ""
        cta = "Learn More"
        has_cta = False
        has_visual = False
        has_subtitle = False

        # Prefer PosterBriefV2 messages for more accurate presence.
        if brief and brief.messages:
            for msg in brief.messages:
                if msg.role == "headline":
                    title = msg.content
                elif msg.role == "subhead":
                    subtitle = msg.content
                    if msg.presence != "omit":
                        has_subtitle = True
                elif msg.role == "cta":
                    cta = msg.content
                    if msg.presence == "required":
                        has_cta = True
            for vs in brief.visual_subjects:
                if vs.presence not in ("omit", "none") and vs.role != "none":
                    has_visual = True
        else:
            # Legacy element-based detection.
            for el in elements:
                if el.id == "title" or el.role == "headline":
                    title = el.content
                elif el.id == "subtitle" or el.role == "subhead":
                    subtitle = el.content
                    if el.presence != "omit":
                        has_subtitle = True
                elif el.id == "cta" or el.role == "cta":
                    cta = el.content
                    if el.presence == "required":
                        has_cta = True
                elif el.type.value == "image" or el.role == "visual_label":
                    if el.presence != "omit":
                        has_visual = True

        # Use ArtDirectionV2 colours when available.
        if state.art_direction is not None:
            cs = state.art_direction.color_system
            primary = cs.background
            secondary = cs.secondary
            accent = cs.accent
            text_color = cs.foreground
        elif style is not None:
            primary = style.primary_color
            secondary = style.secondary_color
            accent = style.accent_color
            text_color = style.text_color
        else:
            primary, secondary, accent, text_color = "#1a1a2e", "#16213e", "#00d4ff", "#ffffff"

        return _build_fallback_html(
            width=state.canvas.width,
            height=state.canvas.height,
            primary=primary,
            secondary=secondary,
            accent=accent,
            text_color=text_color,
            title=title,
            subtitle=subtitle or " ",
            cta=cta,
            has_cta=has_cta,
            has_visual=has_visual,
            has_subtitle=has_subtitle,
        )

    # ── helpers ──

    def _configured_for_llm(self) -> bool:
        return bool(
            self.llm_client.api_key
            and self.llm_client.base_url
            and not self.llm_client.model.startswith("mock-")
        )
