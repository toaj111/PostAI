"""Spatial layout planner — Phase 2 (HTML/CSS output).

Produces a complete, self-contained HTML document that the ``HTMLPainter``
renders via headless Chromium.  No more custom schema — the LLM writes
HTML+CSS directly, which it understands natively.

Every iteration calls the LLM with full context (VLM feedback from the
previous round) so the design improves incrementally.
"""

from __future__ import annotations

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

    Falls back to a simple template-built HTML document when the LLM is
    unavailable.
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

    # ── LLM path ──

    async def _run_llm(self, state: GraphState) -> str:
        if state.content_plan is None or state.style is None:
            raise ValueError("content_plan and style are required before layout planning")
        if not self._configured_for_llm():
            raise LLMCallError("LLM provider is not configured")

        messages = self._build_messages(state)
        # Use _chat_completion directly so we get the raw text output (not
        # parsed through a JSON schema).  The LLM returns a full HTML doc.
        # force_raw=True is critical — the LLM must output free-form HTML
        # text, not a JSON object.  Without it the client sends
        # response_format=json_object and the API returns 400.
        content = await self.llm_client._chat_completion(
            messages=messages,
            response_model=type("_Dummy", (), {}),  # unused by _chat_completion for extraction
            force_raw=True,
        )
        html = self._extract_html(content)
        self._validate_html(html)
        return html

    def _build_messages(self, state: GraphState) -> list[dict]:
        """Build the LLM prompt asking for a complete HTML poster."""
        w = state.canvas.width
        h = state.canvas.height
        style = state.style

        system = (
            "You are a poster designer.  Output a COMPLETE, self-contained "
            "HTML document for a poster.  Use inline CSS or a <style> block. "
            "Do NOT use JavaScript.\n\n"
            "Requirements:\n"
            f"- The <body> IS the poster canvas: exactly {w}x{h}px.  Set "
            f"body {{ width:{w}px; height:{h}px; overflow:hidden; }} in CSS.\n"
            "- Use CSS gradients, shadows (box-shadow, text-shadow), "
            "border-radius, and opacity for modern visual effects.\n"
            "- Load fonts via @import from Google Fonts CDN, e.g. "
            "@import url('https://fonts.googleapis.com/css2?family=...').\n"
            "- For images, use inline SVG or placeholder images "
            "(e.g. https://placehold.co/600x400/EEE/31343C).\n"
            "- Every element from the ContentPlan MUST appear in the poster.\n"
            f"- Use these colours from the StyleGuide: primary={style.primary_color}, "
            f"secondary={style.secondary_color}, accent={style.accent_color}, "
            f"text={style.text_color}, mood={style.mood}.\n"
            "- The design must be visually impressive — a real, polished poster, "
            "not a wireframe or documentation example.\n\n"
            "- If reference images are provided, you MAY insert them via <img> or "
            "CSS background-image and must size/place them carefully: keep each image "
            "roughly within 15%-35% of canvas area, keep safe margins, avoid covering "
            "title/CTA, and use object-fit for proper cropping.\n\n"
            "IMPORTANT: Return ONLY the HTML source, starting with <!DOCTYPE html>. "
            "Do NOT wrap it in markdown code fences."
        )

        # Build feedback context from previous VLM critique.
        feedback_text = ""
        if state.feedback_history:
            latest_fb = state.feedback_history[-1]
            parts = []
            # Include browser console errors so the LLM can fix broken CSS / fonts.
            if state.render_result and state.render_result.console_errors:
                import json
                parts.append(
                    "Browser console errors from your previous HTML: "
                    + json.dumps(state.render_result.console_errors, ensure_ascii=False)
                )
            if state.vision_reasoning:
                parts.append(f"Vision model reasoning: {state.vision_reasoning[:2000]}")
            if latest_fb.vision_description:
                parts.append(f"What the vision model saw: {latest_fb.vision_description}")
            if latest_fb.issues:
                import json
                parts.append(f"Issues found: {json.dumps(latest_fb.issues, ensure_ascii=False)}")
            if latest_fb.suggestions:
                import json
                parts.append(
                    f"Suggested improvements: {json.dumps(latest_fb.suggestions, ensure_ascii=False)}"
                )
            if parts:
                parts.insert(0, "=== Feedback from VLM review of the PREVIOUS version ===")
                parts.append(
                    "Rewrite the HTML to address these specific issues. "
                    "Keep the overall structure and content, but fix the problems "
                    "described above. Output the complete revised HTML document."
                )
                feedback_text = "\n".join(parts)

        user_parts = [
            f"User prompt: {state.user_prompt}",
            f"Canvas: {w}x{h}px",
            f"Content plan: {state.content_plan.model_dump(mode='json')}",
            f"Style guide: {style.model_dump(mode='json')}",
        ]
        if state.reference_images:
            refs = [
                f"{index}. {image.url} | {image.description}"
                for index, image in enumerate(state.reference_images, start=1)
            ]
            user_parts.append(
                "Reference images available for insertion and style context:\n"
                + "\n".join(refs)
            )
        if feedback_text:
            user_parts.append(feedback_text)
        user_parts.append("Output the complete HTML document now.")

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def _extract_html(self, content: str) -> str:
        """Strip markdown code fences if the LLM wraps the HTML."""
        html = content.strip()
        # Remove leading ```html fence
        html = re.sub(r"^```html?\s*\n?", "", html, flags=re.IGNORECASE)
        # Remove trailing ``` fence
        html = re.sub(r"\n?```\s*$", "", html)
        html = html.strip()
        if not html:
            raise SchemaParseError("LLM returned empty HTML after stripping fences")
        return html

    def _validate_html(self, html: str) -> None:
        """Guard against obviously broken HTML before handing it to the Painter.

        The Painter (Playwright) can render almost anything, but we catch
        the most common LLM mistakes early so they trigger a retry instead of
        producing a blank screenshot.
        """
        lower = html.lower()
        if "<body" not in lower and "<html" not in lower:
            raise SchemaParseError("HTML output must contain a <body> or <html> tag")
        if len(html) < 50:
            raise SchemaParseError("HTML output is too short to be a valid poster document")

    # ── fallback ──

    def _run_fallback(self, state: GraphState) -> str:
        """Template-built poster when the LLM is not available."""
        style = state.style
        elements = state.content_plan.elements if state.content_plan else []

        title = "Poster"
        subtitle = ""
        cta = "Learn More"
        for el in elements:
            if el.id == "title":
                title = el.content
            elif el.id == "subtitle":
                subtitle = el.content
            elif el.id == "cta":
                cta = el.content

        return _build_fallback_html(
            width=state.canvas.width,
            height=state.canvas.height,
            primary=style.primary_color if style else "#1a1a2e",
            secondary=style.secondary_color if style else "#16213e",
            accent=style.accent_color if style else "#00d4ff",
            text_color=style.text_color if style else "#ffffff",
            title=title,
            subtitle=subtitle or " ",
            cta=cta,
        )

    # ── helpers ──

    def _configured_for_llm(self) -> bool:
        return bool(
            self.llm_client.api_key
            and self.llm_client.base_url
            and not self.llm_client.model.startswith("mock-")
        )
