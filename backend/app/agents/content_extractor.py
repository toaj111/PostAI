from __future__ import annotations

import re

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.llm_client import StructuredLLMClient
from app.schemas.agents import ContentPlan, ElementContent
from app.schemas.layout import ElementType
from app.schemas.state import GraphState


class ContentExtractor:
    def __init__(self, llm_client: StructuredLLMClient | None = None) -> None:
        settings = get_settings()
        self.allow_model_fallback = settings.allow_model_fallback
        self.llm_client = llm_client or StructuredLLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            response_format=settings.llm_response_format,
        )

    async def run(self, state: GraphState) -> ContentPlan:
        try:
            return await self._run_llm(state)
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_llm() and not self.allow_model_fallback:
                raise
            if self._configured_for_llm():
                state.warnings.append(f"ContentExtractor LLM fallback: {exc}")
            return self._run_rules(state)

    async def _run_llm(self, state: GraphState) -> ContentPlan:
        reference_context = self._build_reference_context(state)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior poster copy planner. Return only JSON matching ContentPlan. "
                    "Create 4 to 8 stable poster elements. Required ids: title, subtitle, main_visual, cta. "
                    "Use type values: text, image, shape, group. Keep Chinese text concise. "
                    "When reference images are provided, use their subject/style as context for planning."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User prompt: {state.user_prompt}\n"
                    f"Canvas: {state.canvas.width}x{state.canvas.height}px\n"
                    f"{reference_context}"
                    "Plan semantic poster content."
                ),
            },
        ]
        plan = await self._parse_content_plan(messages)
        self._validate_required_elements(plan)
        return plan

    async def _parse_content_plan(self, messages: list[dict[str, str]]) -> ContentPlan:
        try:
            return await self.llm_client.parse(messages=messages, response_model=ContentPlan)
        except SchemaParseError:
            if not hasattr(self.llm_client, "_chat_completion"):
                raise
            content = await self.llm_client._chat_completion(messages=messages, response_model=ContentPlan)
            import json

            payload = json.loads(content)
            normalized = self._normalize_content_payload(payload)
            return self.llm_client.validate_payload(normalized, ContentPlan)

    def _run_rules(self, state: GraphState) -> ContentPlan:
        prompt = state.user_prompt.strip()
        normalized = re.sub(r"\s+", " ", prompt)
        title = self._make_title(normalized)
        reference_hint = ""
        if state.reference_images:
            reference_hint = "；参考图线索：" + "；".join(img.description for img in state.reference_images[:2])

        elements = [
            ElementContent(id="title", type=ElementType.text, content=title, priority=10),
            ElementContent(
                id="subtitle",
                type=ElementType.text,
                content=self._make_subtitle(normalized),
                priority=8,
            ),
            ElementContent(
                id="main_visual",
                type=ElementType.image,
                content=f"{normalized} 的核心视觉，干净背景，适合中文海报排版{reference_hint}",
                priority=7,
                alt="poster key visual",
            ),
            ElementContent(
                id="info",
                type=ElementType.text,
                content=self._make_info(normalized),
                priority=6,
            ),
            ElementContent(
                id="cta",
                type=ElementType.text,
                content="立即了解",
                priority=5,
            ),
        ]
        return ContentPlan(elements=elements, poster_goal=f"为「{normalized}」生成一张清晰、有层级的宣传海报")

    def _configured_for_llm(self) -> bool:
        return bool(self.llm_client.api_key and self.llm_client.base_url and not self.llm_client.model.startswith("mock-"))

    def _build_reference_context(self, state: GraphState) -> str:
        if not state.reference_images:
            return "Reference images: none.\n"
        lines = ["Reference images:"]
        for index, image in enumerate(state.reference_images, start=1):
            lines.append(f"{index}. {image.url} | description: {image.description}")
        lines.append("Use these references to infer visual subject and composition priorities.")
        return "\n".join(lines) + "\n"

    def _validate_required_elements(self, plan: ContentPlan) -> None:
        ids = {element.id for element in plan.elements}
        missing = {"title", "subtitle", "main_visual", "cta"} - ids
        if missing:
            raise SchemaParseError(f"Content plan is missing required elements: {sorted(missing)}")

    def _normalize_content_payload(self, payload: dict) -> dict:
        if "poster_goal" in payload and all("content" in item for item in payload.get("elements", [])):
            return payload

        elements = payload.get("elements")
        if not isinstance(elements, list):
            return payload

        normalized_elements: list[dict] = []
        title_seen = False
        subtitle_seen = False
        main_visual_seen = False
        cta_seen = False

        for index, item in enumerate(elements):
            if not isinstance(item, dict):
                continue
            raw_id = str(item.get("id") or f"element_{index}")
            raw_type = str(item.get("type") or "text")
            children = item.get("children") if isinstance(item.get("children"), list) else []
            props = item.get("props") if isinstance(item.get("props"), dict) else {}
            content = item.get("content") or item.get("text") or props.get("text") or props.get("prompt")
            element_type = "text"

            if raw_type == "image":
                element_type = "image"
                content = content or props.get("src") or props.get("alt") or f"{raw_id} poster visual"
            elif raw_type in {"shape", "rect", "circle"}:
                element_type = "shape"
                content = content or f"{raw_id} decorative shape"
            elif raw_type in {"group", "container"} and children:
                child_texts = [
                    str(child.get("text") or child.get("content") or child.get("props", {}).get("text"))
                    for child in children
                    if isinstance(child, dict) and (child.get("text") or child.get("content") or child.get("props", {}).get("text"))
                ]
                content = " ".join(child_texts) or raw_id
                element_type = "text"
            else:
                content = content or raw_id

            mapped_id = raw_id
            if raw_id in {"background", "bg"}:
                continue
            if raw_id == "main_visual":
                main_visual_seen = True
                element_type = "image"
            if raw_id == "title":
                title_seen = True
            if raw_id == "subtitle":
                subtitle_seen = True
            if raw_id == "cta":
                cta_seen = True
            if raw_type == "image" and not main_visual_seen:
                mapped_id = "main_visual"
                main_visual_seen = True
            priority = 10 if mapped_id == "title" else 8 if mapped_id == "subtitle" else 7 if mapped_id == "main_visual" else 5
            normalized_elements.append({"id": mapped_id, "type": element_type, "content": str(content), "priority": priority})

        if not title_seen:
            normalized_elements.insert(0, {"id": "title", "type": "text", "content": "科技风 AI 会议", "priority": 10})
        if not subtitle_seen:
            normalized_elements.append({"id": "subtitle", "type": "text", "content": "探索智能创意与未来视觉体验", "priority": 8})
        if not main_visual_seen:
            normalized_elements.append({"id": "main_visual", "type": "image", "content": "AI conference futuristic key visual", "priority": 7})
        if not cta_seen:
            normalized_elements.append({"id": "cta", "type": "text", "content": "立即了解", "priority": 5})

        return {
            "poster_goal": payload.get("poster_goal") or payload.get("goal") or "生成清晰、有层级的宣传海报",
            "target_audience": payload.get("target_audience"),
            "elements": normalized_elements,
        }

    def _make_title(self, prompt: str) -> str:
        prompt = prompt.replace("制作一张", "").replace("生成一张", "").replace("海报", "").strip()
        return prompt[:24] if prompt else "智能海报设计"

    def _make_subtitle(self, prompt: str) -> str:
        if any(keyword in prompt.lower() for keyword in ["ai", "人工智能", "科技"]):
            return "探索智能创意与未来视觉体验"
        if any(keyword in prompt for keyword in ["音乐", "演出", "节"]):
            return "沉浸现场，释放灵感"
        if any(keyword in prompt for keyword in ["招聘", "校园", "社团"]):
            return "加入我们，一起创造更多可能"
        return "用清晰视觉传达核心信息"

    def _make_info(self, prompt: str) -> str:
        if any(keyword in prompt for keyword in ["会议", "大会", "论坛"]):
            return "2026.06 | Shanghai"
        if any(keyword in prompt for keyword in ["活动", "演出", "展览"]):
            return "周末开放 | 欢迎参与"
        return "PostAI Generated Poster"
