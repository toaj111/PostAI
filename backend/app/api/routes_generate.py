from __future__ import annotations

import base64
import re
from binascii import Error as BinasciiError
from collections.abc import AsyncIterator
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.events import format_sse
from app.orchestration.graph_runner import GraphRunner
from app.core.config import get_settings
from app.core.errors import RenderError
from app.render.asset_store import AssetStore
from app.render.html_painter import HTMLPainter, apply_canvas_guard
from app.agents.html_refiner import HTMLRefiner
from app.schemas.api import (
    GenerateRequest,
    GenerateResponse,
    RefineRequest,
    RefineResponse,
    ReferenceImageUploadRequest,
    ReferenceImageUploadResponse,
)
from app.schemas.layout import CanvasSpec
from app.schemas.state import GraphState


router = APIRouter(tags=["generate"])

_CORE_IDS = {"headline", "key-visual", "cta", "subtitle", "main-visual"}


def _asset_store() -> AssetStore:
    settings = get_settings()
    return AssetStore(settings.asset_dir, settings.asset_url_path)


def _state_from_request(request: GenerateRequest) -> GraphState:
    return GraphState(
        user_prompt=request.prompt,
        canvas=CanvasSpec(width=request.width, height=request.height),
        max_iterations=request.max_iterations,
        min_iterations=request.min_iterations,
        target_score=request.target_score,
        reference_images=request.reference_images,
    )


def _safe_filename(filename: str) -> str:
    parsed = urlparse(filename)
    candidate = parsed.path.rsplit("/", 1)[-1].strip()
    return candidate or "reference-image"


def _html_ids(html: str) -> set[str]:
    return set(re.findall(r"""\bid\s*=\s*["']([^"']+)["']""", html, flags=re.IGNORECASE))


@router.post("/reference-images/upload", response_model=ReferenceImageUploadResponse)
async def upload_reference_image(request: Request, payload: ReferenceImageUploadRequest) -> ReferenceImageUploadResponse:
    try:
        prefix, encoded = payload.data_url.split(",", 1)
        if not prefix.startswith("data:") or ";base64" not in prefix:
            raise ValueError("data_url must be a base64 data URL")
        raw_bytes = base64.b64decode(encoded)
    except (ValueError, BinasciiError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid uploaded image payload: {exc}") from exc

    mime_type = payload.mime_type.strip().lower()
    if not mime_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="uploaded file must be an image/* mime type")

    try:
        asset_url = await _asset_store().save_reference_image(
            raw_bytes,
            filename=_safe_filename(payload.filename),
            mime_type=mime_type,
        )
    except RenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ReferenceImageUploadResponse(
        url=str(request.base_url).rstrip("/") + asset_url,
        filename=_safe_filename(payload.filename),
        mime_type=mime_type,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    runner = GraphRunner()
    try:
        return await runner.run(_state_from_request(request))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/generate/stream")
async def generate_stream(request: GenerateRequest) -> StreamingResponse:
    runner = GraphRunner()
    state = _state_from_request(request)

    async def stream() -> AsyncIterator[str]:
        async for sse_event in runner.run_events(state):
            yield format_sse(sse_event)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════════════
# Refine endpoint (Phase 1-4: incremental HTML micro-adjustment)
# ═══════════════════════════════════════════════════════════════════════════════


@router.post("/refine", response_model=RefineResponse)
async def refine(request: RefineRequest) -> RefineResponse:
    """Refine an existing poster HTML based on a user instruction.

    Reads the current HTML (via *html_url* or *layout_html*), passes it to
    the HTMLRefiner agent together with the user's refinement instruction,
    re-renders the PNG, and returns the updated result.
    """
    store = _asset_store()

    # 1. Read current HTML.
    if request.layout_html:
        current_html = request.layout_html
    elif request.html_url:
        try:
            current_html = await store.load_html_by_url(request.html_url)
        except RenderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=400, detail="Either html_url or layout_html must be provided")

    # 2. Determine iteration number.
    iteration = request.iteration if request.iteration is not None else 1

    # 3. Refine.
    refiner = HTMLRefiner()
    refined_html = await refiner.run(current_html, request.prompt, width=request.width, height=request.height)
    warnings: list[str] = []
    if refined_html.startswith("<!-- REFINE FAILED:"):
        warnings.append("LLM refine failed; returned original HTML unchanged")
        refined_html = current_html
    refined_html = apply_canvas_guard(refined_html, width=request.width, height=request.height)

    # 4. Check core ID preservation (Phase 2).
    missing_ids = sorted((_html_ids(current_html) & _CORE_IDS) - _html_ids(refined_html))
    if missing_ids:
        warnings.append(f"Core element IDs lost during refine: {missing_ids}")

    # 5. Render.
    painter = HTMLPainter()
    render_result = await painter.render(refined_html, width=request.width, height=request.height)

    # 6. Save refined HTML and PNG.
    html_url = await store.save_refined_html(refined_html, job_id=request.job_id, iteration=iteration)
    image_url = await store.save_refined_png(
        render_result.image_base64 or "", job_id=request.job_id, iteration=iteration
    )

    # 7. Phase 4: VLM critique of the refined result.
    critique = None
    try:
        from app.agents.vlm_critic import HeuristicVLMCritic
        from app.schemas.state import GraphState, RenderResult as StateRenderResult
        critic = HeuristicVLMCritic()
        # Build a minimal state for the critic.
        critic_state = GraphState(user_prompt=request.prompt)
        critic_state.layout_html = refined_html
        critic_state.render_result = StateRenderResult(
            image_base64=render_result.image_base64,
            width=request.width, height=request.height,
        )
        critique = await critic.run(critic_state)

        # If the refine made things worse and VLM didn't pass, auto-retry once
        # with the VLM feedback injected into the refinement instruction.
        if not critique.passed and critique.score < 70:
            retry_prompt = (
                f"{request.prompt}\n\n"
                f"[额外反馈] 上一次微调有以下问题：{'; '.join(critique.issues[:3])}。"
                f"修复建议：{'; '.join(critique.suggestions[:3])}。"
                "请重新微调以修复这些问题。"
            )
            refined_html = await refiner.run(current_html, retry_prompt, width=request.width, height=request.height)
            if not refined_html.startswith("<!-- REFINE FAILED:"):
                refined_html = apply_canvas_guard(refined_html, width=request.width, height=request.height)
                render_result = await painter.render(refined_html, width=request.width, height=request.height)
                html_url = await store.save_refined_html(refined_html, job_id=request.job_id, iteration=iteration)
                image_url = await store.save_refined_png(
                    render_result.image_base64 or "", job_id=request.job_id, iteration=iteration
                )
                # Re-critique.
                critic_state.layout_html = refined_html
                critic_state.render_result = StateRenderResult(
                    image_base64=render_result.image_base64,
                    width=request.width, height=request.height,
                )
                critique = await critic.run(critic_state)
    except Exception:
        pass  # VLM critique is best-effort.

    return RefineResponse(
        job_id=request.job_id,
        iteration=iteration,
        layout_html=refined_html,
        html_url=html_url,
        image_url=image_url,
        final_image=render_result.image_base64,
        warnings=warnings,
        critique=critique,
    )
