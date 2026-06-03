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
from html import escape
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
  body {{ width:{width}px; height:{height}px; overflow:hidden; position:relative;
          font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif;
          background:{primary}; color:{text_color}; }}
  .bg {{ position:absolute; inset:0; background:
    linear-gradient(118deg, {primary} 0%, {primary} 52%, {secondary} 52.2%, {secondary} 68%, {primary} 68.4%),
    radial-gradient(circle at 18% 18%, {accent}33 0 17%, transparent 17.4%),
    radial-gradient(circle at 88% 78%, {secondary}88 0 22%, transparent 22.6%);
  }}
  .grain {{ position:absolute; inset:0; opacity:.18; mix-blend-mode:screen;
    background-image:
      repeating-linear-gradient(0deg, rgba(255,255,255,.08) 0 1px, transparent 1px 7px),
      repeating-linear-gradient(90deg, rgba(255,255,255,.04) 0 1px, transparent 1px 11px);
  }}
  .ghost-type {{ position:absolute; left:-6%; top:8%; width:120%; color:{accent};
    font-size:{ghost_size}px; line-height:.82; font-weight:1000; letter-spacing:0;
    opacity:.11; transform:rotate(-7deg); transform-origin:left top;
    overflow:hidden; white-space:normal; word-break:break-all; }}
  .rule {{ position:absolute; height:2px; background:{accent}; opacity:.9; }}
  .rule.a {{ left:8%; right:16%; top:12%; }}
  .rule.b {{ left:20%; right:8%; bottom:14%; }}
  .index {{ position:absolute; right:7%; top:10%; writing-mode:vertical-rl;
    font-size:{meta_size}px; font-weight:800; color:{accent}; letter-spacing:.12em; }}
  .content {{ position:relative; z-index:2; display:flex; flex-direction:column; justify-content:flex-end;
    height:100%; padding:0 8% 18% 8%; }}
  .kicker {{ font-size:{meta_size}px; font-weight:800; color:{accent}; margin-bottom:5%;
    text-transform:uppercase; letter-spacing:.14em; }}
  .title {{ font-size:{title_size}px; font-weight:1000; line-height:.95; max-width:90%;
    text-wrap:balance; text-shadow:0 10px 0 rgba(0,0,0,.16); }}
  .footnote {{ position:absolute; left:8%; bottom:6%; width:48%; font-size:{note_size}px;
    line-height:1.45; opacity:.68; }}
</style>
</head>
<body>
<div class="bg"></div>
<div class="grain"></div>
<div class="ghost-type" aria-hidden="true">{title}</div>
<div class="rule a"></div>
<div class="rule b"></div>
<div class="index">POSTER / 01</div>
<div class="content">
  <div class="kicker">POSTAI GENERATED</div>
  <div id="headline" data-role="headline" class="title">{title}</div>
</div>
<div class="footnote">type, rhythm, scale, contrast</div>
</body>
</html>"""

_FALLBACK_IMAGE_LED = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ width:{width}px; height:{height}px; overflow:hidden; position:relative;
          font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif;
          background:{primary}; color:{text_color}; }}
  .bg {{ position:absolute; inset:0; background:
    radial-gradient(circle at 70% 22%, {accent}44 0 16%, transparent 16.4%),
    radial-gradient(circle at 18% 72%, {secondary}66 0 24%, transparent 24.6%),
    linear-gradient(160deg, {primary} 0%, {secondary} 62%, {primary} 100%);
  }}
  .texture {{ position:absolute; inset:0; opacity:.22;
    background:
      repeating-linear-gradient(115deg, rgba(255,255,255,.12) 0 1px, transparent 1px 13px),
      repeating-linear-gradient(0deg, rgba(0,0,0,.16) 0 1px, transparent 1px 9px);
    mix-blend-mode:overlay;
  }}
  .visual {{ position:absolute; left:-7%; top:4%; width:114%; height:61%;
    display:flex; align-items:center; justify-content:center; overflow:hidden; }}
  .visual::before {{ content:""; position:absolute; width:72%; aspect-ratio:1; border:2px solid {accent};
    border-radius:50%; opacity:.55; transform:translate(8%, -2%) rotate(-12deg); }}
  .visual::after {{ content:""; position:absolute; right:7%; bottom:10%; width:38%; height:20%;
    background:{accent}; opacity:.86; transform:skewX(-18deg) rotate(-8deg); mix-blend-mode:screen; }}
  .visual svg {{ position:relative; width:86%; height:86%; filter:drop-shadow(0 28px 40px rgba(0,0,0,.35)); }}
  .text-zone {{ position:absolute; left:0; right:0; bottom:0; height:39%; padding:7% 8% 7%;
    display:grid; grid-template-columns:1fr .28fr; column-gap:6%; align-items:end;
    background:linear-gradient(180deg, rgba(0,0,0,0), rgba(0,0,0,.28) 18%, rgba(0,0,0,.48)); }}
  .title {{ grid-column:1 / 2; font-size:{title_size}px; font-weight:1000; line-height:.96;
    color:{text_color}; max-width:96%; text-wrap:balance; }}
  .subtitle {{ grid-column:1 / 2; font-size:{subtitle_size}px; color:{text_color};
    opacity:.82; margin-top:5%; line-height:1.45; max-width:90%; }}
  .side-meta {{ grid-column:2 / 3; grid-row:1 / span 2; justify-self:end; align-self:stretch;
    border-left:2px solid {accent}; padding-left:18%; color:{accent}; font-size:{meta_size}px;
    font-weight:800; writing-mode:vertical-rl; letter-spacing:.1em; }}
</style>
</head>
<body>
<div class="bg"></div>
<div class="texture"></div>
<div id="key-visual" data-role="visual" class="visual">
  <svg viewBox="0 0 600 560" aria-hidden="true">
    <defs>
      <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0" stop-color="{accent}"/>
        <stop offset=".52" stop-color="{text_color}" stop-opacity=".72"/>
        <stop offset="1" stop-color="{secondary}"/>
      </linearGradient>
    </defs>
    <path d="M84 348 C135 128 280 54 438 90 C552 116 592 225 546 336 C494 462 334 524 198 476 C110 445 64 421 84 348Z" fill="url(#g)" opacity=".8"/>
    <path d="M124 164 L524 96 L438 488 L48 408 Z" fill="none" stroke="{text_color}" stroke-width="10" opacity=".22"/>
    <path d="M188 98 L494 430" stroke="{accent}" stroke-width="18" opacity=".75"/>
    <circle cx="294" cy="282" r="86" fill="{primary}" opacity=".72"/>
    <circle cx="294" cy="282" r="42" fill="{accent}" opacity=".9"/>
  </svg>
</div>
<div class="text-zone">
  <div id="headline" data-role="headline" class="title">{title}</div>
  <div id="subtitle" data-role="subhead" class="subtitle">{subtitle}</div>
  <div class="side-meta">VISUAL SYSTEM</div>
</div>
</body>
</html>"""

_FALLBACK_EVENT_INFO = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ width:{width}px; height:{height}px; overflow:hidden; position:relative;
          font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif;
          background:{primary}; color:{text_color}; }}
  .bg {{ position:absolute; inset:0; background:
    linear-gradient(90deg, rgba(255,255,255,.09) 1px, transparent 1px) 0 0 / {grid_size}px {grid_size}px,
    linear-gradient(0deg, rgba(255,255,255,.07) 1px, transparent 1px) 0 0 / {grid_size}px {grid_size}px,
    radial-gradient(circle at 85% 18%, {accent}36 0 18%, transparent 18.4%),
    {primary}; }}
  .left-bar {{ position:absolute; left:0; top:0; width:14px; height:100%; background:{accent}; }}
  .diagonal {{ position:absolute; right:-18%; top:28%; width:74%; height:16%; background:{secondary};
    opacity:.9; transform:rotate(-14deg); transform-origin:center; }}
  .stamp {{ position:absolute; right:8%; top:8%; width:24%; aspect-ratio:1; border:3px solid {accent};
    color:{accent}; display:flex; align-items:center; justify-content:center; font-weight:1000;
    font-size:{stamp_size}px; transform:rotate(8deg); opacity:.82; }}
  .content {{ position:relative; z-index:2; display:grid; grid-template-rows:auto auto 1fr auto;
    height:100%; padding:12% 9% 9% 13%; }}
  .eyebrow {{ color:{accent}; font-size:{meta_size}px; font-weight:900; letter-spacing:.14em;
    text-transform:uppercase; margin-bottom:9%; }}
  .title {{ font-size:{title_size}px; font-weight:1000; line-height:.98; margin-bottom:6%;
    max-width:86%; text-wrap:balance; }}
  .info {{ font-size:{subtitle_size}px; opacity:.88; line-height:1.52; max-width:72%;
    border-top:2px solid rgba(255,255,255,.34); padding-top:5%; }}
  .data-row {{ align-self:end; display:grid; grid-template-columns:repeat(3, 1fr); gap:10px;
    margin-bottom:7%; color:{text_color}; }}
  .data-row span {{ border-top:2px solid {accent}; padding-top:9px; font-size:{note_size}px; opacity:.8; }}
  .cta {{ align-self:end; justify-self:start; font-size:{cta_size}px; font-weight:900;
    color:{primary}; background:{accent}; display:inline-block; padding:14px 34px;
    letter-spacing:.06em; text-transform:uppercase; box-shadow:10px 10px 0 {secondary}; }}
</style>
</head>
<body>
<div class="bg"></div>
<div class="left-bar"></div>
<div class="diagonal"></div>
<div class="stamp">INFO</div>
<div class="content">
  <div class="eyebrow">PUBLIC POSTER</div>
  <div id="headline" data-role="headline" class="title">{title}</div>
  <div id="subtitle" data-role="subhead" class="info">{subtitle}</div>
  <div class="data-row" aria-hidden="true"><span>01 / details</span><span>02 / schedule</span><span>03 / entry</span></div>
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
  body {{ width:{width}px; height:{height}px; overflow:hidden; position:relative;
          font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif;
          background:{primary}; color:{text_color}; }}
  .bg {{ position:absolute; inset:0; background:
    radial-gradient(circle at 24% 20%, {accent}4A 0 15%, transparent 15.4%),
    radial-gradient(circle at 92% 82%, {secondary}99 0 28%, transparent 28.6%),
    linear-gradient(145deg, {primary} 0%, {secondary} 58%, {primary} 100%);
  }}
  .mesh {{ position:absolute; inset:-5%; opacity:.2; transform:rotate(-8deg);
    background-image:
      linear-gradient(90deg, rgba(255,255,255,.22) 1px, transparent 1px),
      linear-gradient(0deg, rgba(255,255,255,.18) 1px, transparent 1px);
    background-size:{grid_size}px {grid_size}px;
  }}
  .content {{ position:relative; z-index:2; height:100%; padding:8% 8% 7%;
    display:grid; grid-template-rows:auto 1fr auto; color:{text_color}; }}
  .topline {{ display:flex; justify-content:space-between; align-items:flex-start;
    color:{accent}; font-size:{meta_size}px; font-weight:900; letter-spacing:.1em; }}
  .stage {{ position:relative; display:grid; grid-template-columns:.9fr 1.1fr; align-items:center; gap:4%; }}
  .copy {{ align-self:center; }}
  .title {{ font-size:{title_size}px; font-weight:1000; line-height:.92; max-width:96%;
    text-wrap:balance; margin-bottom:8%; }}
  .subtitle {{ font-size:{subtitle_size}px; opacity:.86; line-height:1.45; max-width:92%;
    border-left:4px solid {accent}; padding-left:5%; }}
  .visual {{ justify-self:end; width:112%; aspect-ratio:1; position:relative;
    display:flex; align-items:center; justify-content:center; overflow:hidden; }}
  .visual::before {{ content:""; position:absolute; inset:10%; background:{accent}; opacity:.82;
    clip-path:polygon(50% 0, 100% 30%, 82% 100%, 14% 88%, 0 22%); }}
  .visual::after {{ content:""; position:absolute; inset:22%; border:3px solid {text_color};
    transform:rotate(16deg); opacity:.38; }}
  .visual svg {{ position:relative; width:72%; height:72%; filter:drop-shadow(0 22px 28px rgba(0,0,0,.32)); }}
  .bottom {{ display:flex; justify-content:space-between; align-items:end; gap:5%; }}
  .cta {{ font-size:{cta_size}px; font-weight:1000; color:{primary}; background:{accent};
    padding:16px 34px; text-align:center; min-width:36%; box-shadow:9px 9px 0 {text_color};
    letter-spacing:.04em; }}
  .micro {{ max-width:38%; font-size:{note_size}px; line-height:1.45; opacity:.72; text-align:right; }}
</style>
</head>
<body>
<div class="bg"></div>
<div class="mesh"></div>
<div class="content">
  <div class="topline"><span>CAMPAIGN</span><span>POSTAI</span></div>
  <div class="stage">
    <div class="copy">
      <div id="headline" data-role="headline" class="title">{title}</div>
      <div id="subtitle" data-role="subhead" class="subtitle">{subtitle}</div>
    </div>
    <div id="key-visual" data-role="visual" class="visual">
      <svg viewBox="0 0 300 300" aria-hidden="true">
        <path d="M48 188 C62 80 150 28 226 72 C282 104 284 206 218 248 C144 295 38 260 48 188Z" fill="{text_color}" opacity=".78"/>
        <path d="M82 104 L250 86 L218 238 L58 222 Z" fill="none" stroke="{primary}" stroke-width="10" opacity=".55"/>
        <circle cx="150" cy="154" r="46" fill="{primary}" opacity=".86"/>
        <path d="M120 154 L144 178 L196 118" fill="none" stroke="{accent}" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
  </div>
  <div class="bottom">
    <div id="cta" data-role="cta" class="cta">{cta}</div>
    <div class="micro">clear action, strong hierarchy, poster-first layout</div>
  </div>
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
    title = escape(title, quote=True)
    subtitle = escape(subtitle, quote=True)
    cta = escape(cta, quote=True)

    title_size = max(30, int(height * 0.064))
    ghost_size = max(96, int(height * 0.16))
    subtitle_size = max(16, int(height * 0.027))
    cta_size = max(16, int(height * 0.025))
    meta_size = max(10, int(height * 0.015))
    note_size = max(10, int(height * 0.014))
    stamp_size = max(28, int(height * 0.07))
    grid_size = max(28, int(min(width, height) * 0.075))

    # Select template based on content characteristics.
    if has_visual and has_cta:
        # Full campaign: image + CTA (original template).
        return _FALLBACK_CTA_CAMPAIGN.format(
            width=width, height=height,
            primary=primary, secondary=secondary, accent=accent, text_color=text_color,
            title=title, subtitle=subtitle, cta=cta,
            title_size=title_size, subtitle_size=subtitle_size, cta_size=cta_size,
            meta_size=meta_size, note_size=note_size, grid_size=grid_size,
        )
    elif has_visual:
        # Image-led poster: visual dominates, text is secondary.
        return _FALLBACK_IMAGE_LED.format(
            width=width, height=height,
            primary=primary, secondary=secondary, accent=accent, text_color=text_color,
            title=title, subtitle=subtitle,
            title_size=title_size, subtitle_size=subtitle_size,
            meta_size=meta_size,
        )
    elif has_cta:
        # Event/info poster: text + CTA, no image needed.
        return _FALLBACK_EVENT_INFO.format(
            width=width, height=height,
            primary=primary, secondary=secondary, accent=accent, text_color=text_color,
            title=title, subtitle=subtitle, cta=cta,
            title_size=title_size, subtitle_size=subtitle_size, cta_size=cta_size,
            meta_size=meta_size, note_size=note_size, stamp_size=stamp_size,
            grid_size=grid_size,
        )
    else:
        # Type-only poster: pure typography, minimal decoration.
        return _FALLBACK_TYPE_ONLY.format(
            width=width, height=height,
            primary=primary, secondary=secondary, accent=accent, text_color=text_color,
            title=title, subtitle=subtitle,
            title_size=title_size, subtitle_size=subtitle_size,
            ghost_size=ghost_size, meta_size=meta_size, note_size=note_size,
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
