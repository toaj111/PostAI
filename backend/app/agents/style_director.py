"""StyleDirector — Phase 3 (ArtDirectionV2).

Produces an ``ArtDirectionV2`` (structured art direction with composition
language, colour system, typography, and imagery treatment), then converts
it to the legacy ``StyleGuide`` so that downstream agents continue to work.

The ``ArtDirectionV2`` is stored in ``state.art_direction`` for the response
and for future agents (LayoutPlanner V2).
"""

from __future__ import annotations

import json

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.llm_client import StructuredLLMClient
from app.schemas.agents import (
    ArtDirectionV2,
    ColorSystem,
    ImagerySpec,
    PosterLanguage,
    StyleGuide,
    TypographySpec,
    art_direction_to_style_guide,
)
from app.schemas.state import GraphState


class StyleDirector:
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

    async def run(self, state: GraphState) -> StyleGuide:
        """Produce a StyleGuide (and store ArtDirectionV2 in state)."""
        try:
            ad = await self._run_llm_v2(state)
            state.art_direction = ad
            guide = art_direction_to_style_guide(ad)
            self._validate_colors(ad)
            return guide
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_llm() and not self.allow_model_fallback:
                raise
            if self._configured_for_llm():
                state.warnings.append(f"StyleDirector LLM fallback: {exc}")
            guide = self._run_rules(state)
            state.art_direction = self._style_guide_to_art_direction(guide, state)
            return guide

    # ── LLM path (ArtDirectionV2) ──

    async def _run_llm_v2(self, state: GraphState) -> ArtDirectionV2:
        """Call the text LLM and ask for an ArtDirectionV2 JSON."""
        messages = self._build_v2_messages(state)
        return await self._parse_art_direction(messages)

    def _build_v2_messages(self, state: GraphState) -> list[dict]:
        reference_context = ""
        if state.reference_images:
            refs = [
                f"{index}. {image.url} | {image.description}"
                for index, image in enumerate(state.reference_images, start=1)
            ]
            reference_context = (
                "\nReference images are provided. Extract useful palette, mood, cropping, "
                "texture, or subject cues, but do not copy them blindly:\n"
                + "\n".join(refs)
            )

        brief_json = state.poster_brief.model_dump(mode="json") if state.poster_brief else None

        system = (
            "You are an art director specializing in posters across editorial, "
            "cultural, commercial, and experimental design.\n"
            "Return only JSON matching ArtDirectionV2.\n\n"
            "Choose a poster language that fits the PosterBriefV2.\n"
            "Do NOT default to generic neon gradients, rounded UI cards, "
            "CTA buttons, or centered landing-page composition.\n\n"
            "The style should describe:\n"
            "- style_name: a short descriptive name for this direction\n"
            "- mood_keywords: 2-4 words capturing the emotional tone\n"
            "- poster_language: composition_family, visual_density, negative_space, "
            "depth_strategy, risk_level\n"
            "- color_system: background, foreground, accent, secondary (all 6-digit HEX), "
            "palette_notes explaining how colour serves the theme\n"
            "- typography: headline_style, body_style, scale_contrast, letter_case\n"
            "- imagery: treatment, background_strategy, prompt, negative_prompt\n\n"
            "Make the art direction poster-specific, not generic. "
            "A typographic minimal poster should have sparse density, generous negative space, "
            "and flat depth. A music festival should be dense, energetic, with diagonal energy. "
            "An art exhibition should feel editorial, expressive, with thoughtful typography.\n\n"
            f"{reference_context}"
        )

        user = (
            f"User prompt: {state.user_prompt}\n\n"
            f"Poster brief:\n{json.dumps(brief_json, ensure_ascii=False) if brief_json else 'none'}\n\n"
            "Produce an ArtDirectionV2 that gives the layout planner a strong "
            "poster-specific direction."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def _parse_art_direction(self, messages: list[dict]) -> ArtDirectionV2:
        """Parse LLM response as ArtDirectionV2, with fallback normalisation."""
        try:
            return await self.llm_client.parse(messages=messages, response_model=ArtDirectionV2)
        except SchemaParseError:
            if not hasattr(self.llm_client, "_chat_completion"):
                raise
            content = await self.llm_client._chat_completion(
                messages=messages, response_model=StyleGuide
            )
            payload = json.loads(content)
            normalized = self._normalize_ad_payload(payload)
            return self.llm_client.validate_payload(normalized, ArtDirectionV2)

    def _normalize_ad_payload(self, payload: dict) -> dict:
        """Normalize a raw LLM JSON response into a valid ArtDirectionV2 shape."""
        # If the LLM returned old StyleGuide format, convert it.
        if "primary_color" in payload and "poster_language" not in payload:
            return {
                "style_name": payload.get("mood", "modern"),
                "mood_keywords": payload.get("theme_keywords", []),
                "poster_language": {
                    "composition_family": "centered_iconic",
                    "visual_density": "medium",
                    "negative_space": "balanced",
                    "depth_strategy": "flat",
                    "risk_level": "safe",
                },
                "color_system": {
                    "background": payload.get("primary_color", "#111111"),
                    "foreground": payload.get("text_color", "#FFFFFF"),
                    "accent": payload.get("accent_color", "#FF0000"),
                    "secondary": payload.get("secondary_color", "#333333"),
                    "palette_notes": "",
                },
                "typography": {
                    "headline_style": "grotesk",
                    "body_style": "sans",
                    "scale_contrast": "high",
                    "letter_case": "as_given",
                },
                "imagery": {
                    "treatment": "none",
                    "background_strategy": "gradient",
                    "prompt": payload.get("background_prompt", ""),
                    "negative_prompt": payload.get("negative_prompt", ""),
                },
            }

        if "poster_language" not in payload:
            payload["poster_language"] = {}
        if "color_system" not in payload:
            payload["color_system"] = {}
        if "typography" not in payload:
            payload["typography"] = {}
        if "imagery" not in payload:
            payload["imagery"] = {}
        return payload

    # ── validation ──

    def _validate_colors(self, ad: ArtDirectionV2) -> None:
        cs = ad.color_system
        for field_name in ("background", "foreground", "accent", "secondary"):
            value = getattr(cs, field_name)
            if not isinstance(value, str) or not _is_hex_color(value):
                raise SchemaParseError(f"color_system.{field_name} must be a 6-digit HEX color")

    # ── deterministic fallback (poster_type + tone template table) ──

    def _run_rules(self, state: GraphState) -> StyleGuide:
        """Fallback: select a style based on poster_type and tone, not just keywords."""
        brief = state.poster_brief
        poster_type = brief.poster_intent.poster_type if brief else "custom"
        tones = brief.poster_intent.tone if brief else []
        prompt_lower = state.user_prompt.lower()

        # ── template table keyed by poster_type ──
        if poster_type == "artistic" or "minimal" in tones:
            return self._make_style(
                theme=["minimal", "gallery", "refined"],
                bg="clean off-white gallery wall texture with generous negative space",
                neg="clutter, heavy gradients, rounded UI cards, neon effects",
                primary="#F5F1EB", secondary="#2C2C2C", accent="#C41E3A",
                text_color="#1A1A1A", font="sans-serif", mood="minimal",
            )
        if poster_type == "typographic" or "experimental" in tones:
            return self._make_style(
                theme=["typographic", "editorial", "bold"],
                bg="plain high-contrast field, solid single colour background",
                neg="busy texture, photographic backgrounds, decorative flourishes",
                primary="#0D0D0D", secondary="#E5E5E5", accent="#FF3366",
                text_color="#FAFAFA", font="sans-serif", mood="experimental",
            )
        if poster_type == "exhibition":
            return self._make_style(
                theme=["gallery", "spatial", "cultured"],
                bg="museum wall texture, soft architectural lighting, deep shadow edges",
                neg="neon glow, crowded composition, cartoonish elements",
                primary="#1C1C1C", secondary="#8B7E74", accent="#D4A574",
                text_color="#F5F0E8", font="serif", mood="cultured",
            )
        if poster_type == "event" or poster_type == "campaign":
            if any(kw in prompt_lower for kw in ["音乐", "演出", "live", "festival", "节"]):
                return self._make_style(
                    theme=["music", "energy", "stage"],
                    bg="vibrant stage light gradient with dark readable top area",
                    neg="overcrowded crowd, messy text",
                    primary="#151019", secondary="#FF477E", accent="#FFD166",
                    text_color="#FFF8F0", font="display", mood="energetic",
                )
            return self._make_style(
                theme=["dynamic", "event", "bold"],
                bg="deep gradient field with directional light sweep, clear text zone",
                neg="static composition, low contrast, boring symmetry",
                primary="#0B1026", secondary="#1E88E5", accent="#00E5FF",
                text_color="#F8FBFF", font="sans-serif", mood="dynamic",
            )
        if poster_type == "recruitment":
            return self._make_style(
                theme=["campus", "fresh", "friendly"],
                bg="bright clean campus inspired background with soft geometric shapes",
                neg="low contrast, clutter",
                primary="#F7FAFC", secondary="#2B6CB0", accent="#38A169",
                text_color="#1A202C", font="sans-serif", mood="friendly",
            )
        if poster_type == "product":
            return self._make_style(
                theme=["product", "clean", "premium"],
                bg="studio-lit product plinth on clean neutral background, subtle soft shadow",
                neg="busy patterns, cluttered layout, low-end feel",
                primary="#F8F9FA", secondary="#212529", accent="#D4A574",
                text_color="#212529", font="sans-serif", mood="premium",
            )
        if poster_type == "informational":
            return self._make_style(
                theme=["structured", "clear", "data"],
                bg="clean grid-backed field with organised zones",
                neg="abstract decoration, low readability, overlapping elements",
                primary="#FAFBFC", secondary="#1A1A2E", accent="#2563EB",
                text_color="#111827", font="sans-serif", mood="professional",
            )

        # Fallback to keyword-based detection (legacy).
        if any(kw in prompt_lower for kw in ["ai", "科技", "tech", "未来"]):
            return self._make_style(
                theme=["technology", "neon", "clean"],
                bg="deep navy futuristic gradient with subtle light grid and clean center space",
                neg="busy texture, unreadable text, clutter",
                primary="#0B1026", secondary="#1E88E5", accent="#00E5FF",
                text_color="#F8FBFF", font="sans-serif", mood="futuristic",
            )
        if any(kw in prompt_lower for kw in ["音乐", "演出", "live", "festival"]):
            return self._make_style(
                theme=["music", "energy", "stage"],
                bg="vibrant stage light gradient with dark readable top area",
                neg="overcrowded crowd, messy text",
                primary="#151019", secondary="#FF477E", accent="#FFD166",
                text_color="#FFF8F0", font="display", mood="energetic",
            )
        if any(kw in prompt_lower for kw in ["招聘", "校园", "社团"]):
            return self._make_style(
                theme=["campus", "fresh", "friendly"],
                bg="bright clean campus inspired background with soft geometric shapes",
                neg="low contrast, clutter",
                primary="#F7FAFC", secondary="#2B6CB0", accent="#38A169",
                text_color="#1A202C", font="sans-serif", mood="friendly",
            )
        return self._make_style(
            theme=["minimal", "balanced", "poster"],
            bg="clean modern abstract background with clear readable text area",
            neg="busy details, low contrast",
            primary="#202124", secondary="#4F46E5", accent="#F59E0B",
            text_color="#FFFFFF", font="sans-serif", mood="modern",
        )

    @staticmethod
    def _make_style(**kwargs) -> StyleGuide:
        return StyleGuide(
            theme_keywords=kwargs["theme"],
            background_prompt=kwargs["bg"],
            negative_prompt=kwargs["neg"],
            primary_color=kwargs["primary"],
            secondary_color=kwargs["secondary"],
            accent_color=kwargs["accent"],
            text_color=kwargs["text_color"],
            font_family=kwargs["font"],
            mood=kwargs["mood"],
        )

    def _style_guide_to_art_direction(self, guide: StyleGuide, state: GraphState) -> ArtDirectionV2:
        """Convert a legacy StyleGuide into ArtDirectionV2 (best-effort)."""
        brief = state.poster_brief
        poster_type = brief.poster_intent.poster_type if brief else "custom"

        # Infer composition family from mood.
        comp_map = {
            "futuristic": "centered_iconic",
            "energetic": "diagonal_energy",
            "friendly": "centered_iconic",
            "minimal": "minimal",
            "experimental": "typographic",
            "cultured": "editorial_spread",
            "dynamic": "diagonal_energy",
            "premium": "centered_iconic",
            "professional": "swiss_grid",
            "modern": "minimal",
        }

        return ArtDirectionV2(
            style_name=f"{poster_type} {guide.mood}",
            mood_keywords=list(guide.theme_keywords),
            poster_language=PosterLanguage(
                composition_family=comp_map.get(guide.mood, "centered_iconic"),
                visual_density="dense" if guide.mood == "energetic" else "medium",
                negative_space="generous" if guide.mood in ("minimal", "cultured") else "balanced",
                depth_strategy="layered" if guide.mood in ("futuristic", "energetic") else "flat",
                risk_level="expressive" if guide.mood in ("experimental", "energetic") else "safe",
            ),
            color_system=ColorSystem(
                background=guide.primary_color,
                foreground=guide.text_color,
                accent=guide.accent_color,
                secondary=guide.secondary_color,
                palette_notes="",
            ),
            typography=TypographySpec(
                headline_style="grotesk",
                body_style="sans",
                scale_contrast="high",
                letter_case="as_given",
            ),
            imagery=ImagerySpec(
                treatment="none",
                background_strategy="gradient",
                prompt=guide.background_prompt,
                negative_prompt=guide.negative_prompt or "",
            ),
        )

    def _configured_for_llm(self) -> bool:
        return bool(self.llm_client.api_key and self.llm_client.base_url and not self.llm_client.model.startswith("mock-"))


def _is_hex_color(value: str) -> bool:
    return len(value) == 7 and value.startswith("#") and all(char in "0123456789abcdefABCDEF" for char in value[1:])
