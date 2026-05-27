from __future__ import annotations

import base64
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
from app.schemas.api import (
    GenerateRequest,
    GenerateResponse,
    ReferenceImageUploadRequest,
    ReferenceImageUploadResponse,
)
from app.schemas.layout import CanvasSpec
from app.schemas.state import GraphState


router = APIRouter(tags=["generate"])


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
