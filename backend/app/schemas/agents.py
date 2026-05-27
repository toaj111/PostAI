from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.layout import ElementType


class ElementContent(BaseModel):
    id: str = Field(description="Stable element id, such as title or main_visual")
    type: ElementType
    content: str
    priority: int = Field(default=5, ge=1, le=10)
    alt: str | None = None
    role: str | None = Field(
        default=None,
        description="Semantic role: headline, subhead, body, cta, date, venue, visual_label, etc.",
    )
    presence: str = Field(
        default="required",
        description="required | recommended | optional | omit — controls whether the element must appear",
    )


class ContentPlan(BaseModel):
    elements: list[ElementContent] = Field(default_factory=list)
    poster_goal: str
    target_audience: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — PosterBriefV2: structured poster-editor brief
# ═══════════════════════════════════════════════════════════════════════════════


class PosterMessage(BaseModel):
    """A single text unit on the poster (headline, date, CTA, etc.)."""

    id: str
    role: str = Field(
        default="body",
        description="headline | subhead | body | meta | date | venue | price | logo | sponsor | cta | caption | credit | visual_label",
    )
    content: str
    importance: int = Field(default=5, ge=1, le=10)
    presence: str = Field(default="required", description="required | recommended | optional")
    source: str = Field(default="user", description="user | inferred | placeholder")
    editable: bool = True
    notes: str | None = None


class VisualSubject(BaseModel):
    """A visual element on the poster (photo, illustration, shape, etc.)."""

    id: str
    role: str = Field(
        default="illustration",
        description="photo | illustration | symbol | texture | pattern | shape | frame | ornament | none",
    )
    description: str
    presence: str = Field(default="required", description="required | recommended | optional | omit")
    source: str = Field(default="user", description="user | reference | inferred")
    avoid: list[str] = Field(default_factory=list)


class PosterIntent(BaseModel):
    """What kind of poster this is and what it should achieve."""

    poster_type: str = Field(
        default="custom",
        description="event | exhibition | campaign | editorial | announcement | recruitment | product | typographic | artistic | informational | custom",
    )
    communication_mode: str = Field(
        default="inform",
        description="announce | invite | inform | persuade | evoke | provoke | celebrate | sell",
    )
    primary_goal: str
    target_audience: str | None = None
    tone: list[str] = Field(default_factory=list)


class ContentStrategy(BaseModel):
    """Rules that govern what content should or should not appear."""

    headline_policy: str = Field(
        default="literal",
        description="literal | poetic | minimal | no_headline",
    )
    information_density: str = Field(default="medium", description="sparse | medium | dense")
    cta_policy: str = Field(default="omit", description="required | optional | omit")
    image_policy: str = Field(
        default="optional", description="required | optional | omit | reference_driven"
    )
    inference_policy: str = "do_not_invent_specific_facts"


class PosterBriefV2(BaseModel):
    """Phase 2 structured poster brief — the content blueprint.

    Replaces the flat ContentPlan element list with a design-aware brief that
    captures poster intent, content strategy, and visual subject decisions.
    """

    poster_intent: PosterIntent
    content_strategy: ContentStrategy = Field(default_factory=ContentStrategy)
    messages: list[PosterMessage] = Field(default_factory=list)
    visual_subjects: list[VisualSubject] = Field(default_factory=list)
    must_not_do: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Compatibility converter — PosterBriefV2 ↔ ContentPlan
# ═══════════════════════════════════════════════════════════════════════════════


def poster_brief_to_content_plan(brief: PosterBriefV2) -> ContentPlan:
    """Convert a PosterBriefV2 into the legacy ContentPlan format.

    Used so that downstream agents (StyleDirector, LayoutPlanner, VLM critic)
    continue to work without any changes.
    """
    elements: list[ElementContent] = []

    for msg in brief.messages:
        el_type = ElementType.image if msg.role == "visual_label" else ElementType.text
        elements.append(
            ElementContent(
                id=msg.id,
                type=el_type,
                content=msg.content,
                priority=msg.importance,
                role=msg.role,
                presence=msg.presence,
            )
        )

    for vs in brief.visual_subjects:
        if vs.role == "none" or vs.presence == "omit":
            continue
        elements.append(
            ElementContent(
                id=vs.id,
                type=ElementType.image,
                content=vs.description,
                priority=7,
                role="visual_label",
                presence=vs.presence,
            )
        )

    return ContentPlan(
        elements=elements,
        poster_goal=brief.poster_intent.primary_goal,
        target_audience=brief.poster_intent.target_audience,
    )


def content_plan_to_poster_brief(plan: ContentPlan) -> PosterBriefV2:
    """Convert a legacy ContentPlan back into PosterBriefV2 (best-effort).

    Useful for the fallback path where ContentPlan is generated directly.
    """
    messages: list[PosterMessage] = []
    visual_subjects: list[VisualSubject] = []

    for el in plan.elements:
        role = el.role or (
            "headline" if el.id == "title" else
            "subhead" if el.id == "subtitle" else
            "cta" if el.id == "cta" else
            "visual_label" if el.type == ElementType.image else
            "body"
        )
        if role in ("visual_label",) or el.type == ElementType.image:
            visual_subjects.append(
                VisualSubject(
                    id=el.id,
                    role="illustration",
                    description=el.content,
                    presence=el.presence,
                    source="inferred",
                )
            )
        else:
            messages.append(
                PosterMessage(
                    id=el.id,
                    role=role,
                    content=el.content,
                    importance=el.priority,
                    presence=el.presence,
                    source="inferred",
                )
            )

    # Detect cta_policy from presence of CTA
    has_cta = any(m.role == "cta" and m.presence == "required" for m in messages)
    cta_policy = "required" if has_cta else "omit"

    return PosterBriefV2(
        poster_intent=PosterIntent(
            poster_type="custom",
            communication_mode="inform",
            primary_goal=plan.poster_goal,
            target_audience=plan.target_audience,
        ),
        content_strategy=ContentStrategy(cta_policy=cta_policy),
        messages=messages,
        visual_subjects=visual_subjects,
        must_not_do=["不要凭空编造精确日期地点", "不要默认加入报名按钮"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — ArtDirectionV2: structured art direction for posters
# ═══════════════════════════════════════════════════════════════════════════════


class PosterLanguage(BaseModel):
    """Composition and spatial decisions that define the poster's visual language."""

    composition_family: str = Field(
        default="centered_iconic",
        description="swiss_grid | centered_iconic | diagonal_energy | editorial_spread | collage | typographic | cinematic | brutalist | minimal | ornamental | custom",
    )
    visual_density: str = Field(default="medium", description="sparse | medium | dense")
    negative_space: str = Field(
        default="balanced",
        description="generous | balanced | tight | intentionally_crowded",
    )
    depth_strategy: str = Field(
        default="flat", description="flat | layered | photographic | 3d | print_texture"
    )
    risk_level: str = Field(default="safe", description="safe | expressive | experimental")


class ColorSystem(BaseModel):
    background: str = "#111111"
    foreground: str = "#F6F1E8"
    accent: str = "#E83F3F"
    secondary: str = "#4A90E2"
    palette_notes: str = ""


class TypographySpec(BaseModel):
    headline_style: str = Field(
        default="grotesk",
        description="condensed_bold | elegant_serif | grotesk | handwritten | monospace | custom",
    )
    body_style: str = Field(default="sans", description="sans | serif | mono | none")
    scale_contrast: str = Field(default="high", description="low | medium | high | extreme")
    letter_case: str = Field(
        default="as_given", description="as_given | uppercase | lowercase | mixed"
    )


class ImagerySpec(BaseModel):
    treatment: str = Field(
        default="none",
        description="none | photo_crop | illustration | symbol | abstract_geometry | texture | reference_image",
    )
    background_strategy: str = Field(
        default="gradient",
        description="plain | gradient | image_full_bleed | split_field | pattern | paper | custom",
    )
    prompt: str = ""
    negative_prompt: str = ""


class ArtDirectionV2(BaseModel):
    """Phase 3 structured art direction — the visual blueprint.

    Replaces the flat StyleGuide with a design-aware brief that captures
    composition language, colour system, typography strategy, and imagery
    treatment.
    """

    style_name: str = ""
    mood_keywords: list[str] = Field(default_factory=list)
    poster_language: PosterLanguage = Field(default_factory=PosterLanguage)
    color_system: ColorSystem = Field(default_factory=ColorSystem)
    typography: TypographySpec = Field(default_factory=TypographySpec)
    imagery: ImagerySpec = Field(default_factory=ImagerySpec)


# ═══════════════════════════════════════════════════════════════════════════════
# Compatibility converter — ArtDirectionV2 ↔ StyleGuide
# ═══════════════════════════════════════════════════════════════════════════════


def art_direction_to_style_guide(ad: ArtDirectionV2) -> StyleGuide:
    """Convert an ArtDirectionV2 into the legacy StyleGuide format."""
    font_map = {
        "condensed_bold": "display",
        "elegant_serif": "serif",
        "grotesk": "sans-serif",
        "handwritten": "cursive",
        "monospace": "monospace",
        "custom": "sans-serif",
    }
    return StyleGuide(
        theme_keywords=list(ad.mood_keywords),
        background_prompt=ad.imagery.prompt or f"{ad.color_system.background} {ad.imagery.background_strategy} background",
        negative_prompt=ad.imagery.negative_prompt or None,
        primary_color=ad.color_system.background,
        secondary_color=ad.color_system.secondary,
        accent_color=ad.color_system.accent,
        text_color=ad.color_system.foreground,
        font_family=font_map.get(ad.typography.headline_style, "sans-serif"),
        mood=ad.mood_keywords[0] if ad.mood_keywords else "modern",
    )


class StyleGuide(BaseModel):
    theme_keywords: list[str] = Field(default_factory=list)
    background_prompt: str
    negative_prompt: str | None = None
    primary_color: str
    secondary_color: str
    accent_color: str
    text_color: str
    font_family: str = "sans-serif"
    mood: str


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — CritiqueResultV2: structured poster critique
# ═══════════════════════════════════════════════════════════════════════════════


class CritiqueIssue(BaseModel):
    """A single concrete issue found in the rendered poster."""

    type: str = Field(default="composition", description="composition | typography | content | color | imagery | rendering | style")
    severity: str = Field(default="minor", description="minor | major | blocking")
    target_id: str | None = Field(default=None, description="Element id this issue relates to, e.g. 'headline'")
    description: str = ""
    suggestion: str = ""


class CritiqueRubric(BaseModel):
    """Dimension-level scores summing to the total score (max 20 each, total 120 normalised)."""

    poster_identity: int = Field(default=0, ge=0, le=20, description="Does it feel like a finished poster?")
    topic_fit: int = Field(default=0, ge=0, le=20, description="Does it express the user's theme?")
    composition: int = Field(default=0, ge=0, le=20, description="Clear visual idea and hierarchy?")
    typography: int = Field(default=0, ge=0, le=20, description="Is type intentional and suitable?")
    readability: int = Field(default=0, ge=0, le=20, description="Can required information be read?")
    craft: int = Field(default=0, ge=0, le=20, description="No broken rendering, overlap, or generic template feel?")


class CritiqueResult(BaseModel):
    """Phase 5 critique — structured poster review.

    Backward-compatible with Phase 2: the legacy ``issues`` (str list) and
    ``suggestions`` (str list) fields are kept alongside the new structured
    ``structured_issues``, ``rubric``, and ``revision_focus``.
    """

    score: int = Field(ge=0, le=100)
    passed: bool
    reasoning: str
    vision_description: str = Field(
        default="",
        description="Literal description of what the vision model sees in the rendered poster",
    )
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(
        default_factory=list,
        description="Natural-language actionable suggestions for the layout planner",
    )
    # ── Phase 5 V2 fields ──
    structured_issues: list[CritiqueIssue] = Field(default_factory=list)
    rubric: CritiqueRubric | None = None
    revision_focus: str = Field(
        default="layout",
        description="final | layout | style | content | render — drives the router",
    )
    do_not_change: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_vlm_score_shape(cls, data):
        """Accept common VLM mistake: putting rubric object in ``score``.

        The intended shape is ``score: int`` plus ``rubric: {...}``, but vision
        models often return ``score`` as the rubric object. Normalising here
        avoids repeated expensive VLM retries and preserves the useful rubric.
        """
        if not isinstance(data, dict):
            return data

        score_payload = data.get("score")
        if not isinstance(score_payload, dict):
            return data

        rubric_fields = (
            "poster_identity",
            "topic_fit",
            "composition",
            "typography",
            "readability",
            "craft",
        )
        if data.get("rubric") is None:
            data["rubric"] = {key: score_payload.get(key, 0) for key in rubric_fields}

        total = 0
        for key in rubric_fields:
            try:
                total += int(score_payload.get(key, 0))
            except (TypeError, ValueError):
                total += 0
        data["score"] = max(0, min(100, round(total / 120 * 100)))
        return data

