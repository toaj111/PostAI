"""Tests for the incremental HTML refine pipeline (Phase 1-4)."""

import pytest

from app.agents.html_refiner import HTMLRefiner
from app.render.asset_store import AssetStore
from app.render.html_painter import _build_fallback_html
from app.core.errors import RenderError

_SAMPLE_HTML = """<!DOCTYPE html>
<html><head><style>
  body { width:512px; height:768px; overflow:hidden; font-family:sans-serif; }
  #headline { font-size:40px; color:#fff; position:absolute; top:80px; left:60px; }
  #key-visual { position:absolute; top:200px; left:50%; transform:translateX(-50%); }
  #cta { position:absolute; bottom:60px; left:50%; transform:translateX(-50%); }
</style></head><body>
<div id="headline" data-role="headline">Headline</div>
<div id="key-visual" data-role="visual"><svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="none" stroke="white" stroke-width="2"/></svg></div>
<div id="cta" data-role="cta">Buy Now</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — HTMLRefiner agent
# ═══════════════════════════════════════════════════════════════════════════════


class TestHTMLRefiner:
    def test_html_refiner_extract_html_strips_fences(self):
        """_extract_html strips markdown code fences."""
        refiner = HTMLRefiner()
        result = refiner._extract_html("```html\n<div>hello</div>\n```")
        assert result == "<div>hello</div>"

    def test_html_refiner_validate_html_accepts_valid(self):
        refiner = HTMLRefiner()
        refiner._validate_html("<html><body><h1>Hello World, This is a Poster Title</h1></body></html>")

    def test_html_refiner_validate_html_rejects_too_short(self):
        refiner = HTMLRefiner()
        with pytest.raises(Exception):
            refiner._validate_html("<html><body>Hi</body></html>")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — AssetStore HTML loading
# ═══════════════════════════════════════════════════════════════════════════════


class TestAssetStoreLoadHTML:
    async def test_load_html_by_url_returns_saved_content(self, tmp_path):
        store = AssetStore(str(tmp_path), "/assets")
        html = "<html><body>test</body></html>"
        url = await store.save_html(html, job_id="abc", iteration=1)
        loaded = await store.load_html_by_url(url)
        assert loaded == html

    async def test_load_html_by_url_rejects_path_traversal(self, tmp_path):
        store = AssetStore(str(tmp_path), "/assets")
        with pytest.raises(RenderError, match="unsafe"):
            await store.load_html_by_url("/assets/../../../etc/passwd.html")

    async def test_load_html_by_url_rejects_non_html(self, tmp_path):
        store = AssetStore(str(tmp_path), "/assets")
        with pytest.raises(RenderError, match=".html"):
            await store.load_html_by_url("/assets/image.png")

    async def test_load_html_by_url_rejects_wrong_prefix(self, tmp_path):
        store = AssetStore(str(tmp_path), "/assets")
        with pytest.raises(RenderError, match="start with"):
            await store.load_html_by_url("http://evil.com/stolen.html")

    async def test_load_html_by_url_rejects_missing_file(self, tmp_path):
        store = AssetStore(str(tmp_path), "/assets")
        with pytest.raises(RenderError, match="not found"):
            await store.load_html_by_url("/assets/nonexistent.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Element ID preservation
# ═══════════════════════════════════════════════════════════════════════════════


class TestElementIDPreservation:
    """Verify that HTMLRefiner preserves stable element IDs."""

    def test_sample_html_has_core_ids(self):
        """The sample HTML used for tests must have the required core IDs."""
        assert 'id="headline"' in _SAMPLE_HTML
        assert 'id="key-visual"' in _SAMPLE_HTML
        assert 'id="cta"' in _SAMPLE_HTML

    def test_fallback_html_has_core_ids(self):
        """Fallback template should carry element IDs (Phase 2)."""
        html = _build_fallback_html(
            width=512, height=768,
            title="Test", subtitle="Sub", cta="Go",
            has_cta=True, has_visual=True, has_subtitle=True,
        )
        # The cta_campaign template should have headline, key-visual, and cta IDs.
        assert 'id="headline"' in html
        assert 'id="key-visual"' in html
        assert 'id="cta"' in html


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Patch operations
# ═══════════════════════════════════════════════════════════════════════════════


class TestPatchApplier:
    def test_apply_patch_adds_css_rule_to_style_block(self):
        refiner = HTMLRefiner()
        ops = [{"op": "set_css", "selector": "#headline", "property": "top", "value": "120px"}]
        result = refiner._apply_patch(_SAMPLE_HTML, ops)
        assert "#headline { top: 120px; }" in result

    def test_apply_patch_handles_multiple_ops(self):
        refiner = HTMLRefiner()
        ops = [
            {"op": "set_css", "selector": "#headline", "property": "top", "value": "90px"},
            {"op": "set_css", "selector": "#key-visual", "property": "transform", "value": "scale(1.2)"},
        ]
        result = refiner._apply_patch(_SAMPLE_HTML, ops)
        assert "#headline { top: 90px; }" in result
        assert "#key-visual { transform: scale(1.2); }" in result

    def test_apply_patch_ignores_non_css_ops(self):
        refiner = HTMLRefiner()
        ops = [{"op": "add_element", "selector": "#cta"}]
        result = refiner._apply_patch(_SAMPLE_HTML, ops)
        assert result == _SAMPLE_HTML  # No change.

    def test_apply_patch_no_style_block_adds_before_body_close(self):
        html = "<!DOCTYPE html><html><body><div id='headline'>Hi</div></body></html>"
        refiner = HTMLRefiner()
        ops = [{"op": "set_css", "selector": "#headline", "property": "color", "value": "red"}]
        result = refiner._apply_patch(html, ops)
        assert "<style>" in result
        assert "#headline { color: red; }" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Refined HTML & PNG persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestRefinedPersistence:
    async def test_save_refined_html(self, tmp_path):
        store = AssetStore(str(tmp_path), "/assets")
        url = await store.save_refined_html("<html></html>", job_id="job1", iteration=2)
        assert url.startswith("/assets/")
        assert "refine_2.html" in url

    async def test_save_refined_png(self, tmp_path):
        import base64
        store = AssetStore(str(tmp_path), "/assets")
        png_base64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").decode()
        url = await store.save_refined_png(png_base64, job_id="job1", iteration=1)
        assert url.startswith("/assets/")
        assert "refine_1.png" in url
