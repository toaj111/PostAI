from __future__ import annotations

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.llm_client import StructuredLLMClient
from app.schemas.agents import StyleGuide
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

    async def run(self, state: GraphState) -> StyleGuide:
        try:
            return await self._run_llm(state)
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_llm() and not self.allow_model_fallback:
                raise
            if self._configured_for_llm():
                state.warnings.append(f"StyleDirector LLM fallback: {exc}")
            return self._run_rules(state)

    async def _run_llm(self, state: GraphState) -> StyleGuide:
        content_plan = state.content_plan.model_dump(mode="json") if state.content_plan else None
        reference_context = ""
        if state.reference_images:
            refs = [
                f"{index}. {image.url} | {image.description}"
                for index, image in enumerate(state.reference_images, start=1)
            ]
            reference_context = (
                "\nReference images are provided. Extract palette, mood, and compositional cues from them "
                "while keeping user prompt as the top priority:\n"
                + "\n".join(refs)
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an art director for poster design. Return only JSON matching StyleGuide. "
                    "All colors must be valid 6-digit HEX values. "
                    "The background prompt must reserve clean readable areas for text."
                    f"{reference_context}"
                ),
            },
            {"role": "user", "content": f"User prompt: {state.user_prompt}\nContent plan: {content_plan}"},
        ]
        guide = await self.llm_client.parse(messages=messages, response_model=StyleGuide)
        self._validate_hex_colors(guide)
        return guide

    def _run_rules(self, state: GraphState) -> StyleGuide:
        prompt = state.user_prompt.lower()
        if any(keyword in prompt for keyword in ["ai", "科技", "tech", "未来"]):
            return StyleGuide(
                theme_keywords=["technology", "neon", "clean"],
                background_prompt="deep navy futuristic gradient with subtle light grid and clean center space",
                negative_prompt="busy texture, unreadable text, clutter",
                primary_color="#0B1026",
                secondary_color="#1E88E5",
                accent_color="#00E5FF",
                text_color="#F8FBFF",
                font_family="tech",
                mood="futuristic",
            )
        if any(keyword in prompt for keyword in ["音乐", "演出", "live", "festival"]):
            return StyleGuide(
                theme_keywords=["music", "energy", "stage"],
                background_prompt="vibrant stage light gradient with dark readable top area",
                negative_prompt="overcrowded crowd, messy text",
                primary_color="#151019",
                secondary_color="#FF477E",
                accent_color="#FFD166",
                text_color="#FFF8F0",
                font_family="display",
                mood="energetic",
            )
        if any(keyword in prompt for keyword in ["招聘", "校园", "社团"]):
            return StyleGuide(
                theme_keywords=["campus", "fresh", "friendly"],
                background_prompt="bright clean campus inspired background with soft geometric shapes",
                negative_prompt="low contrast, clutter",
                primary_color="#F7FAFC",
                secondary_color="#2B6CB0",
                accent_color="#38A169",
                text_color="#1A202C",
                font_family="sans-serif",
                mood="friendly",
            )
        return StyleGuide(
            theme_keywords=["minimal", "balanced", "poster"],
            background_prompt="clean modern abstract background with clear readable text area",
            negative_prompt="busy details, low contrast",
            primary_color="#202124",
            secondary_color="#4F46E5",
            accent_color="#F59E0B",
            text_color="#FFFFFF",
            font_family="sans-serif",
            mood="modern",
        )

    def _configured_for_llm(self) -> bool:
        return bool(self.llm_client.api_key and self.llm_client.base_url and not self.llm_client.model.startswith("mock-"))

    def _validate_hex_colors(self, guide: StyleGuide) -> None:
        for field_name in ("primary_color", "secondary_color", "accent_color", "text_color"):
            value = getattr(guide, field_name)
            if not isinstance(value, str) or not _is_hex_color(value):
                raise SchemaParseError(f"{field_name} must be a 6-digit HEX color")


def _is_hex_color(value: str) -> bool:
    return len(value) == 7 and value.startswith("#") and all(char in "0123456789abcdefABCDEF" for char in value[1:])
