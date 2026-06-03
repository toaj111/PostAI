from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.schemas.agents import (
    ArtDirectionV2,
    ContentExpansionPlan,
    ContentPlan,
    CritiqueResult,
    PosterBriefV2,
    StyleGuide,
    VisualSystemPlan,
)
from app.schemas.layout import LayoutTree
from app.schemas.state import ReferenceImage, RenderResult


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    width: int = Field(default=1024, ge=256, le=4096)
    height: int = Field(default=1536, ge=256, le=4096)
    max_iterations: int = Field(default=3, ge=1, le=5)
    min_iterations: int = Field(default=0, ge=0, le=4, description="Minimum VLM review cycles before early exit; 0=stop anytime, 1=at least one re-layout")
    target_score: int = Field(default=85, ge=1, le=100)
    reference_images: list[ReferenceImage] = Field(
        default_factory=list,
        max_length=5,
        description="Optional references used as layout/style context and embeddable image URLs",
    )


class ReferenceImageUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=5, max_length=100)
    data_url: str = Field(min_length=20)
    description: str = Field(default="", max_length=500)


class ReferenceImageUploadResponse(BaseModel):
    url: str
    filename: str
    mime_type: str


class GenerateResponse(BaseModel):
    job_id: str
    final_image: str | None
    image_url: str | None = None
    score: int | None = None
    warnings: list[str] = Field(default_factory=list)
    content_plan: ContentPlan | None = None
    poster_brief: PosterBriefV2 | None = None
    content_expansion: ContentExpansionPlan | None = None
    art_direction: ArtDirectionV2 | None = None
    visual_system: VisualSystemPlan | None = None
    style: StyleGuide | None = None
    layout_tree: LayoutTree | None = None
    layout_html: str | None = None
    html_url: str | None = None
    render_result: RenderResult | None = None
    critiques: list[CritiqueResult] = Field(default_factory=list)


# ── Phase 1: incremental HTML refinement ──


class RefineRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    prompt: str = Field(min_length=1, max_length=1000)
    width: int = Field(default=1024, ge=256, le=4096)
    height: int = Field(default=1536, ge=256, le=4096)
    html_url: str | None = None
    layout_html: str | None = None
    iteration: int | None = Field(default=None, ge=0, le=999)

    @model_validator(mode="after")
    def require_html_source(self) -> "RefineRequest":
        if not self.html_url and not self.layout_html:
            raise ValueError("Either html_url or layout_html must be provided")
        return self


class RefineResponse(BaseModel):
    job_id: str
    iteration: int
    layout_html: str
    html_url: str
    image_url: str | None = None
    final_image: str | None = None
    warnings: list[str] = Field(default_factory=list)
    critique: CritiqueResult | None = None
