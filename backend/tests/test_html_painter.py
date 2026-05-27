"""Tests for HTMLPainter — Phase 1."""

import base64

import pytest

from app.core.errors import RenderError
from app.render.html_painter import HTMLPainter, _build_fallback_html, apply_canvas_guard

_MINIMAL_HTML = "<!DOCTYPE html><html><body>Hello</body></html>"
_STYLED_HTML = """<!DOCTYPE html>
<html><head><style>
  body { width:400px; height:600px; background:linear-gradient(180deg, #1a1a2e, #16213e); }
  h1 { color:white; font-family:sans-serif; text-align:center; padding-top:40%; }
</style></head><body><h1>AI Poster</h1></body></html>"""


# ── _build_fallback_html ──


def test_fallback_html_contains_expected_content():
    html = _build_fallback_html(
        width=400,
        height=600,
        title="Hello World",
        subtitle="A test poster",
        cta="Click Here",
        has_cta=True,
        has_visual=True,
        has_subtitle=True,
    )
    assert "Hello World" in html
    assert "A test poster" in html
    assert "Click Here" in html
    assert "400px" in html
    assert "600px" in html


def test_fallback_type_only_template_no_cta_no_visual():
    """Type-only template: no CTA button, no image placeholder."""
    html = _build_fallback_html(
        width=400, height=600,
        title="Pure Type",
        subtitle="",
        has_cta=False,
        has_visual=False,
        has_subtitle=False,
    )
    assert "Pure Type" in html
    # No cta CSS class or CTA text in type-only template.
    assert ".cta" not in html
    assert ".visual" not in html


def test_fallback_image_led_template():
    """Image-led template: visual zone + title, no CTA."""
    html = _build_fallback_html(
        width=400, height=600,
        title="Image Poster",
        subtitle="A visual story",
        has_cta=False,
        has_visual=True,
        has_subtitle=True,
    )
    assert "Image Poster" in html
    assert "A visual story" in html
    assert "text-zone" in html  # image-led has separate text zone


def test_fallback_event_info_template():
    """Event info template: title + info + CTA bar, no image."""
    html = _build_fallback_html(
        width=400, height=600,
        title="Event",
        subtitle="June 2026",
        cta="Register",
        has_cta=True,
        has_visual=False,
        has_subtitle=True,
    )
    assert "Event" in html
    assert "Register" in html
    assert "left-bar" in html


def test_fallback_html_uses_provided_colors():
    html = _build_fallback_html(
        width=400, height=600,
        primary="#FF0000", secondary="#00FF00", accent="#0000FF", text_color="#FFFFFF",
    )
    assert "#FF0000" in html
    assert "#00FF00" in html
    assert "#0000FF" in html


def test_canvas_guard_overrides_responsive_body_scale():
    html = """<!DOCTYPE html><html><head><style>
@media (max-width: 768px) { body { transform: scale(0.5); } }
</style></head><body>Poster</body></html>"""

    guarded = apply_canvas_guard(html, width=768, height=1152)

    assert "postai-canvas-guard" in guarded
    assert "transform: none !important" in guarded
    assert "width: 768px !important" in guarded
    assert guarded.index("postai-canvas-guard") > guarded.index("@media")


# ── HTMLPainter.render ──


async def test_render_minimal_html():
    """Basic HTML renders to a non-empty base64 PNG."""
    result = await HTMLPainter().render(_MINIMAL_HTML, width=400, height=600)
    assert result.image_base64
    assert result.width == 400
    assert result.height == 600
    assert result.mime_type == "image/png"
    # Decode to verify it's a real PNG.
    png_bytes = base64.b64decode(result.image_base64)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic header


async def test_render_styled_html():
    """CSS-styled HTML renders successfully."""
    result = await HTMLPainter().render(_STYLED_HTML, width=400, height=600)
    assert result.image_base64
    assert result.width == 400
    assert result.height == 600


async def test_render_respects_dimensions():
    """PNG dimensions match the requested width/height."""
    result = await HTMLPainter().render(_MINIMAL_HTML, width=800, height=1200)
    assert result.width == 800
    assert result.height == 1200


async def test_render_different_sizes_produce_different_output():
    """Two renders at different sizes should differ."""
    small = await HTMLPainter().render(_MINIMAL_HTML, width=200, height=300)
    large = await HTMLPainter().render(_MINIMAL_HTML, width=800, height=1200)
    assert small.image_base64 != large.image_base64


async def test_render_raises_on_empty_html():
    """Empty HTML must raise RenderError immediately."""
    with pytest.raises(RenderError, match="empty"):
        await HTMLPainter().render("", width=400, height=600)


async def test_render_raises_on_whitespace_only():
    with pytest.raises(RenderError, match="empty"):
        await HTMLPainter().render("   \n  ", width=400, height=600)
