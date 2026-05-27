"""HTMLRefiner — Phase 1 incremental HTML micro-adjustment.

Takes the current poster HTML, a user refinement instruction, and canvas
dimensions, then returns a complete revised HTML document that preserves the
poster's existing concept while applying the requested changes.

Phase 2 adds stable element ID preservation.
Phase 3 adds patch-mode (structured CSS-only changes).
"""

from __future__ import annotations

import json
import re

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.llm_client import StructuredLLMClient


class HTMLRefiner:
    """Refine an existing poster HTML based on a user instruction.

    Uses the text LLM to produce a revised complete HTML document (not a
    patch).  The prompt is designed to minimise drift while applying the
    specific change the user requested.
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

    async def run(self, current_html: str, prompt: str, *, width: int, height: int) -> str:
        """Return a complete revised HTML document."""
        try:
            return await self._run_llm(current_html, prompt, width=width, height=height)
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_llm() and not self.allow_model_fallback:
                raise
            # When LLM is unavailable, return the original HTML with a warning
            # embedded as an HTML comment — the caller should check warnings.
            return f"<!-- REFINE FAILED: {exc} -->\n{current_html}"

    # ── LLM path ──

    async def _run_llm(self, current_html: str, prompt: str, *, width: int, height: int) -> str:
        if not self._configured_for_llm():
            raise LLMCallError("LLM provider is not configured")

        # Truncate HTML to avoid enormous prompts.
        html_snippet = current_html if len(current_html) <= 12000 else (
            current_html[:4000] + "\n... [truncated] ...\n" + current_html[-4000:]
        )

        system = (
            "You are a senior poster HTML/CSS refinement editor.\n"
            "You will receive the current poster HTML and a user's refinement "
            "instruction.\n\n"
            "Your task:\n"
            "- Modify the existing HTML/CSS to satisfy the refinement instruction.\n"
            "- Preserve the poster's existing concept, content, palette, and layout "
            "unless the user explicitly asks to change them.\n"
            "- Make the smallest useful change.\n"
            "- Do NOT add CTA, date, venue, price, sponsor, QR code, or extra copy "
            "unless the user explicitly asks.\n"
            "- Keep all existing required text.\n"
            "- Keep the document self-contained.\n"
            "- Do NOT use JavaScript.\n"
            "- Do NOT add responsive @media rules, viewport scaling, zoom, or "
            "transform: scale(...) on html/body/the poster root.\n"
            f"- html and body must remain exactly {width}px by {height}px with "
            "overflow hidden.\n\n"
            "- IMPORTANT: Preserve all element IDs (e.g. id=\"headline\", "
            "id=\"key-visual\", id=\"cta\") and data-role attributes exactly as "
            "they appear in the current HTML. These are stable identifiers used "
            "for future refinements.\n\n"
            "Return ONLY the complete revised HTML, starting with <!DOCTYPE html>.\n"
            "Do NOT wrap it in markdown code fences."
        )

        user = (
            f"Canvas: {width}x{height}px\n\n"
            f"User refinement instruction:\n{prompt}\n\n"
            f"Current HTML:\n{html_snippet}\n\n"
            "Return the complete refined HTML now."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        content = await self.llm_client._chat_completion(
            messages=messages,
            response_model=type("_Dummy", (), {}),
            force_raw=True,
        )
        html = self._extract_html(content)
        self._validate_html(html)
        return html

    # ── patch mode (Phase 3) ──

    async def run_patch(self, current_html: str, prompt: str, *, width: int, height: int) -> str:
        """Return a complete revised HTML by applying a structured CSS patch.

        The LLM outputs a JSON patch plan; we apply it to the current HTML
        via CSS selector matching.
        """
        try:
            patch = await self._plan_patch(current_html, prompt, width=width, height=height)
            return self._apply_patch(current_html, patch)
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_llm() and not self.allow_model_fallback:
                raise
            # Fall back to full regeneration.
            return await self.run(current_html, prompt, width=width, height=height)

    async def _plan_patch(self, current_html: str, prompt: str, *, width: int, height: int) -> list[dict]:
        """Ask the LLM to output a JSON array of patch operations."""
        html_snippet = current_html if len(current_html) <= 12000 else (
            current_html[:4000] + "\n... [truncated] ...\n" + current_html[-4000:]
        )

        system = (
            "You are a CSS patch planner for poster HTML.\n"
            "Given a poster HTML and a refinement instruction, output a JSON array "
            "of CSS-only patch operations.\n\n"
            "Each operation must have:\n"
            '  - "op": always "set_css"\n'
            '  - "selector": a CSS selector targeting a single element by id, '
            'e.g. "#headline", "#key-visual", "#cta"\n'
            '  - "property": the CSS property name (top, left, font-size, margin-top, '
            'transform, opacity, color, background, etc.)\n'
            '  - "value": the new CSS value as a string\n\n'
            "Rules:\n"
            "- Only change what the user asked to change.\n"
            "- Prefer adjusting existing properties over adding new selectors.\n"
            "- Do NOT add CTA, copy, or new elements.\n"
            "- Keep changes minimal and surgical.\n\n"
            'Return ONLY a JSON array, e.g.:\n'
            '[{"op": "set_css", "selector": "#headline", "property": "top", "value": "420px"}]'
        )

        user = (
            f"Canvas: {width}x{height}px\n\n"
            f"User refinement instruction:\n{prompt}\n\n"
            f"Current HTML:\n{html_snippet}\n\n"
            "Output the JSON patch array now."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        content = await self.llm_client._chat_completion(
            messages=messages,
            response_model=type("_Dummy", (), {}),
            force_raw=True,
        )
        content = content.strip()
        # Strip markdown fences.
        content = re.sub(r"^```(?:json)?\s*\n?", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\n?```\s*$", "", content)
        return json.loads(content)

    def _apply_patch(self, html: str, ops: list[dict]) -> str:
        """Apply CSS patch operations to the HTML string.

        Finds the style block or inline styles for each selector and modifies
        or adds the requested property.
        """
        result = html
        for op in ops:
            if op.get("op") != "set_css":
                continue
            selector = op.get("selector", "")
            prop = op.get("property", "")
            value = op.get("value", "")
            if not selector or not prop:
                continue

            # Build the CSS rule to inject.
            rule = f"{selector} {{ {prop}: {value}; }}"

            # If there's already a <style> block, append the rule.
            if "</style>" in result:
                result = result.replace("</style>", f"\n/* refine patch */\n{rule}\n</style>", 1)
            elif "<style>" in result:
                result = result.replace("<style>", f"<style>\n/* refine patch */\n{rule}\n", 1)
            elif "<style " in result:
                import re as _re
                result = _re.sub(r"(<style[^>]*>)", rf"\1\n/* refine patch */\n{rule}\n", result, count=1)
            else:
                # No style block — insert one before </head> or </body>.
                if "</head>" in result:
                    result = result.replace("</head>", f"<style>\n{rule}\n</style>\n</head>", 1)
                else:
                    result = result.replace("</body>", f"<style>\n{rule}\n</style>\n</body>", 1)

        return result

    # ── helpers ──

    def _extract_html(self, content: str) -> str:
        html = content.strip()
        html = re.sub(r"^```html?\s*\n?", "", html, flags=re.IGNORECASE)
        html = re.sub(r"\n?```\s*$", "", html)
        html = html.strip()
        if not html:
            raise SchemaParseError("LLM returned empty HTML after stripping fences")
        return html

    def _validate_html(self, html: str) -> None:
        lower = html.lower()
        if "<body" not in lower and "<html" not in lower:
            raise SchemaParseError("HTML output must contain a <body> or <html> tag")
        if len(html) < 50:
            raise SchemaParseError("HTML output is too short to be a valid poster document")

    def _configured_for_llm(self) -> bool:
        return bool(
            self.llm_client.api_key
            and self.llm_client.base_url
            and not self.llm_client.model.startswith("mock-")
        )
