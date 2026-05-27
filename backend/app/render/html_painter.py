"""HTMLPainter — render HTML+CSS to a PNG poster via headless Chromium.

Replaces the Pillow-based Painter.  The AI outputs a complete HTML document
with inline CSS; Playwright opens it in headless Chromium and captures a
pixel-perfect screenshot.  No more geometric-primitive limitations.

Uses Playwright's **synchronous** API inside ``asyncio.to_thread`` so that
browser launch runs via plain ``subprocess.Popen``, avoiding uv-managed
Python's broken ``asyncio.create_subprocess_exec`` on Windows.

Usage::

    result = await HTMLPainter().render("<h1>Hello</h1>", width=800, height=600)
"""

from __future__ import annotations

import asyncio
import base64
import re

from playwright.sync_api import sync_playwright

from app.core.errors import RenderError
from app.schemas.state import RenderResult

_CANVAS_GUARD_ID = "postai-canvas-guard"


def apply_canvas_guard(html: str, *, width: int, height: int) -> str:
    """Inject fixed-canvas CSS so responsive rules cannot shrink the poster.

    LLMs sometimes add ``@media`` rules such as ``body { transform: scale(.5) }``.
    Because the browser viewport is exactly the poster width, those media
    queries can fire during screenshot capture and leave most of the PNG blank.
    This guard is appended after model CSS and persisted with generated HTML.
    """
    if _CANVAS_GUARD_ID in html:
        return html

    guard = f"""<style id="{_CANVAS_GUARD_ID}">
  html, body {{
    width: {width}px !important;
    min-width: {width}px !important;
    max-width: {width}px !important;
    height: {height}px !important;
    min-height: {height}px !important;
    max-height: {height}px !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    transform: none !important;
    transform-origin: top left !important;
  }}
  body {{
    position: relative !important;
  }}
</style>"""

    if re.search(r"</head\s*>", html, flags=re.IGNORECASE):
        return re.sub(r"</head\s*>", guard + "\n</head>", html, count=1, flags=re.IGNORECASE)
    return guard + "\n" + html

# ═══════════════════════════════════════════════════════════════════════════════
# Fallback HTML templates — used when the LLM fails to produce valid HTML.
# Four templates replace the old single-CTA layout so the fallback respects
# the poster type instead of always producing a marketing card.
# ═══════════════════════════════════════════════════════════════════════════════

_FALLBACK_TYPE_ONLY = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ width:{width}px; height:{height}px; overflow:hidden; font-family:"Microsoft YaHei","PingFang SC",sans-serif; }}
  .bg {{ position:absolute; inset:0; background:{primary}; }}
  .content {{ position:relative; display:flex; flex-direction:column; justify-content:center;
             align-items:flex-start; height:100%; padding:10% 8%; color:{text_color}; }}
  .title {{ font-size:{title_size}px; font-weight:900; line-height:1.15; max-width:85%; }}
  .accent {{ position:absolute; top:0; right:0; width:35%; height:100%;
             background:{accent}; opacity:0.12; }}
</style>
</head>
<body>
<div class="bg"></div>
<div class="accent"></div>
<div class="content">
  <div id="headline" data-role="headline" class="title">{title}</div>
</div>
</body>
</html>"""

_FALLBACK_IMAGE_LED = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ width:{width}px; height:{height}px; overflow:hidden; font-family:"Microsoft YaHei","PingFang SC",sans-serif; }}
  .bg {{ position:absolute; inset:0; background:linear-gradient(180deg, {primary} 0%, {secondary} 100%); }}
  .visual {{ position:absolute; top:0; left:0; width:100%; height:65%;
             background:rgba(255,255,255,0.05); display:flex; align-items:center; justify-content:center; }}
  .visual svg {{ width:30%; height:30%; opacity:0.3; }}
  .text-zone {{ position:absolute; bottom:0; left:0; width:100%; height:35%;
               display:flex; flex-direction:column; justify-content:center; padding:0 8%; }}
  .title {{ font-size:{title_size}px; font-weight:900; color:{text_color}; }}
  .subtitle {{ font-size:{subtitle_size}px; color:{text_color}; opacity:0.7; margin-top:2%; }}
</style>
</head>
<body>
<div class="bg"></div>
<div id="key-visual" data-role="visual" class="visual">
  <svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="none" stroke="white" stroke-width="1.5" opacity="0.4"/></svg>
</div>
<div class="text-zone">
  <div id="headline" data-role="headline" class="title">{title}</div>
  <div id="subtitle" data-role="subhead" class="subtitle">{subtitle}</div>
</div>
</body>
</html>"""

_FALLBACK_EVENT_INFO = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ width:{width}px; height:{height}px; overflow:hidden; font-family:"Microsoft YaHei","PingFang SC",sans-serif; }}
  .bg {{ position:absolute; inset:0; background:{primary}; }}
  .left-bar {{ position:absolute; left:0; top:0; width:8px; height:100%; background:{accent}; }}
  .content {{ position:relative; display:flex; flex-direction:column; justify-content:center;
             height:100%; padding:8% 10% 8% 12%; color:{text_color}; }}
  .title {{ font-size:{title_size}px; font-weight:900; line-height:1.2; margin-bottom:4%; }}
  .info {{ font-size:{subtitle_size}px; opacity:0.8; line-height:1.6; margin-bottom:6%; }}
  .cta {{ font-size:{cta_size}px; font-weight:700; color:{primary}; background:{accent};
         display:inline-block; padding:12px 36px; }}
</style>
</head>
<body>
<div class="bg"></div>
<div class="left-bar"></div>
<div class="content">
  <div id="headline" data-role="headline" class="title">{title}</div>
  <div id="subtitle" data-role="subhead" class="info">{subtitle}</div>
  <div id="cta" data-role="cta" class="cta">{cta}</div>
</div>
</body>
</html>"""

_FALLBACK_CTA_CAMPAIGN = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ width:{width}px; height:{height}px; overflow:hidden; font-family:"Microsoft YaHei","PingFang SC",sans-serif; }}
  .bg {{ position:absolute; inset:0; background:linear-gradient(180deg, {primary}, {secondary}); }}
  .content {{ position:relative; display:flex; flex-direction:column; justify-content:center;
             align-items:center; height:100%; padding:8% 10%; color:{text_color}; }}
  .title {{ font-size:{title_size}px; font-weight:900; text-align:center; margin-bottom:3%; }}
  .subtitle {{ font-size:{subtitle_size}px; opacity:0.85; text-align:center; margin-bottom:8%; }}
  .visual {{ width:60%; aspect-ratio:1; border-radius:24px; background:rgba(255,255,255,0.08);
             border:2px solid rgba(255,255,255,0.15); display:flex; align-items:center; justify-content:center;
             margin-bottom:8%; }}
  .visual svg {{ width:40%; height:40%; opacity:0.5; }}
  .cta {{ font-size:{cta_size}px; font-weight:700; color:{primary}; background:{accent};
         padding:14px 48px; border-radius:16px; text-align:center; }}
</style>
</head>
<body>
<div class="bg"></div>
<div class="content">
  <div id="headline" data-role="headline" class="title">{title}</div>
  <div id="subtitle" data-role="subhead" class="subtitle">{subtitle}</div>
  <div id="key-visual" data-role="visual" class="visual">
    <svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="none" stroke="white" stroke-width="2"/>
    <circle cx="35" cy="40" r="5" fill="white"/><circle cx="65" cy="40" r="5" fill="white"/>
    <path d="M35 65 Q50 80 65 65" fill="none" stroke="white" stroke-width="3" stroke-linecap="round"/></svg>
  </div>
  <div id="cta" data-role="cta" class="cta">{cta}</div>
</div>
</body>
</html>"""


def _build_fallback_html(
    *,
    width: int,
    height: int,
    primary: str = "#1a1a2e",
    secondary: str = "#16213e",
    accent: str = "#00d4ff",
    text_color: str = "#ffffff",
    title: str = "Poster Title",
    subtitle: str = "Subtitle goes here",
    cta: str = "Learn More",
    has_cta: bool = True,
    has_visual: bool = True,
    has_subtitle: bool = True,
) -> str:
    """Produce a fallback poster, selecting a template that fits the content plan.

    Four templates are available:
    - type_only:       headline + accent block, no image, no CTA
    - image_led:       full-bleed image zone + title/subtitle below
    - event_info:      title + info + CTA, left accent bar, no image
    - cta_campaign:    original centered layout with visual + CTA button
    """
    title_size = int(height * 0.06)
    subtitle_size = int(height * 0.028)
    cta_size = int(height * 0.026)

    # Select template based on content characteristics.
    if has_visual and has_cta:
        # Full campaign: image + CTA (original template).
        return _FALLBACK_CTA_CAMPAIGN.format(
            width=width, height=height,
            primary=primary, secondary=secondary, accent=accent, text_color=text_color,
            title=title, subtitle=subtitle, cta=cta,
            title_size=title_size, subtitle_size=subtitle_size, cta_size=cta_size,
        )
    elif has_visual:
        # Image-led poster: visual dominates, text is secondary.
        return _FALLBACK_IMAGE_LED.format(
            width=width, height=height,
            primary=primary, secondary=secondary, accent=accent, text_color=text_color,
            title=title, subtitle=subtitle,
            title_size=title_size, subtitle_size=subtitle_size,
        )
    elif has_cta:
        # Event/info poster: text + CTA, no image needed.
        return _FALLBACK_EVENT_INFO.format(
            width=width, height=height,
            primary=primary, secondary=secondary, accent=accent, text_color=text_color,
            title=title, subtitle=subtitle, cta=cta,
            title_size=title_size, subtitle_size=subtitle_size, cta_size=cta_size,
        )
    else:
        # Type-only poster: pure typography, minimal decoration.
        return _FALLBACK_TYPE_ONLY.format(
            width=width, height=height,
            primary=primary, secondary=secondary, accent=accent, text_color=text_color,
            title=title, subtitle=subtitle,
            title_size=title_size, subtitle_size=subtitle_size,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HTMLPainter
# ═══════════════════════════════════════════════════════════════════════════════


class HTMLPainter:
    """Render an HTML string to a base64-encoded PNG via headless Chromium.

    The HTML must be self-contained (inline CSS or a ``<style>`` block).
    External resources like Google Fonts are supported — the painter waits
    for ``networkidle`` before capturing.
    """

    async def render(self, html: str, *, width: int, height: int) -> RenderResult:
        """Return a ``RenderResult`` containing the base64 PNG.

        Offloads Playwright's synchronous API to a thread so that the browser
        process is launched via ``subprocess.Popen`` instead of asyncio's
        (often broken on Windows) ``create_subprocess_exec``.

        Raises ``RenderError`` if the HTML is empty or the browser fails.
        """
        sanitised = html.strip()
        if not sanitised:
            raise RenderError("HTML string is empty — nothing to render")
        guarded_html = apply_canvas_guard(sanitised, width=width, height=height)

        def _render_sync() -> RenderResult:
            console_errors: list[str] = []
            with sync_playwright() as p:
                browser = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
                try:
                    page = browser.new_page(viewport={"width": width, "height": height})

                    def _on_console(msg):
                        if msg.type in {"error", "warning"}:
                            console_errors.append(f"[{msg.type}] {msg.text}")

                    page.on("console", _on_console)
                    page.set_content(guarded_html, wait_until="networkidle")
                    screenshot = page.screenshot(full_page=False)
                except Exception as exc:
                    raise RenderError(f"Playwright rendering failed: {exc}") from exc
                finally:
                    browser.close()

            encoded = base64.b64encode(screenshot).decode("ascii")
            return RenderResult(
                image_base64=encoded,
                width=width,
                height=height,
                mime_type="image/png",
                console_errors=console_errors,
            )

        return await asyncio.to_thread(_render_sync)
