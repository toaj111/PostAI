"""ContentExtractor — Phase 2 (PosterBriefV2).

Produces a ``PosterBriefV2`` (structured poster-editor brief with intent,
content strategy, messages, and visual subjects), then converts it to the
legacy ``ContentPlan`` so that downstream agents continue to work.

The ``PosterBriefV2`` is stored in ``state.poster_brief`` for the response
and for future agents (StyleDirector V2, LayoutPlanner V2).
"""

from __future__ import annotations

import json
import re

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.llm_client import StructuredLLMClient
from app.schemas.agents import (
    ContentPlan,
    ContentStrategy,
    ElementContent,
    PosterBriefV2,
    PosterIntent,
    PosterMessage,
    VisualSubject,
    content_plan_to_poster_brief,
    poster_brief_to_content_plan,
)
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

    # ── public API ──

    async def run(self, state: GraphState) -> ContentPlan:
        """Produce a ContentPlan (and store PosterBriefV2 in state).

        The returned ContentPlan keeps the pipeline compatible; the
        PosterBriefV2 on state is the richer, forward-looking brief.
        """
        try:
            brief = await self._run_llm_v2(state)
            state.poster_brief = brief
            plan = poster_brief_to_content_plan(brief)
            self._validate_brief(brief)
            return plan
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_llm() and not self.allow_model_fallback:
                raise
            if self._configured_for_llm():
                state.warnings.append(f"ContentExtractor LLM fallback: {exc}")
            plan = self._run_rules(state)
            state.poster_brief = content_plan_to_poster_brief(plan)
            return plan

    # ── LLM path (PosterBriefV2) ──

    async def _run_llm_v2(self, state: GraphState) -> PosterBriefV2:
        """Call the text LLM and ask for a PosterBriefV2 JSON."""
        reference_context = self._build_reference_context(state)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior poster editor and content strategist.\n"
                    "Return only JSON matching PosterBriefV2.\n\n"
                    "Your job is not to force a marketing template. Decide what kind "
                    "of poster the user is asking for. A poster may be typographic, "
                    "image-led, information-dense, abstract, editorial, event-based, "
                    "product-focused, recruitment-oriented, or purely artistic.\n\n"
                    "Do NOT require title, subtitle, main_visual, or CTA by default. "
                    "Create only the content units that serve the poster intent.\n"
                    "CTA is required only when the user explicitly asks for "
                    "registration, purchase, booking, contact, QR code, call-to-action, "
                    "or when the poster_type/communication_mode clearly needs conversion.\n\n"
                    "Do NOT invent precise factual details such as dates, locations, "
                    "prices, speakers, URLs, or sponsors unless the user provides them. "
                    "If a useful detail is missing, either omit it or mark it as "
                    'placeholder (source="placeholder").\n\n'
                    "For every message set: id, role, content, importance (1-10), "
                    'presence (required|recommended|optional), source (user|inferred|placeholder), '
                    "editable, and notes (why it is needed or can be omitted).\n\n"
                    "For every visual subject set: id, role (photo|illustration|symbol|"
                    "texture|pattern|shape|frame|ornament|none), description, presence, "
                    "source (user|reference|inferred), and avoid list.\n\n"
                    "Set poster_intent with poster_type, communication_mode, primary_goal, "
                    "target_audience, and tone.\n"
                    "Set content_strategy with headline_policy, information_density, "
                    "cta_policy, image_policy, and inference_policy.\n\n"
                    "Reference images, when provided, are visual context. They can "
                    "influence subject, mood, palette, and composition, but user intent "
                    "remains the priority. Put reference image insights into "
                    "visual_subjects with source='reference'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User prompt: {state.user_prompt}\n"
                    f"Canvas: {state.canvas.width}x{state.canvas.height}px\n"
                    f"{reference_context}"
                    "Build a PosterBriefV2 that preserves the user's intent and "
                    "leaves unnecessary poster elements out."
                ),
            },
        ]
        return await self._parse_poster_brief(messages)

    async def _parse_poster_brief(self, messages: list[dict[str, str]]) -> PosterBriefV2:
        """Parse LLM response as PosterBriefV2, with fallback normalisation."""
        try:
            return await self.llm_client.parse(messages=messages, response_model=PosterBriefV2)
        except SchemaParseError:
            if not hasattr(self.llm_client, "_chat_completion"):
                raise
            content = await self.llm_client._chat_completion(
                messages=messages, response_model=ContentPlan
            )
            payload = json.loads(content)
            normalized = self._normalize_brief_payload(payload)
            return self.llm_client.validate_payload(normalized, PosterBriefV2)

    def _normalize_brief_payload(self, payload: dict) -> dict:
        """Normalize a raw LLM JSON response into a valid PosterBriefV2 shape.

        Handles the case where the LLM returns the old ContentPlan format
        or a malformed PosterBriefV2.
        """
        # If the LLM returned the old ContentPlan format, convert it.
        if "elements" in payload and "poster_intent" not in payload:
            plan_elements = payload.get("elements", [])
            messages = []
            visual_subjects = []
            has_cta = False
            for item in plan_elements:
                if not isinstance(item, dict):
                    continue
                rid = str(item.get("id", ""))
                rtype = str(item.get("type", "text"))
                props = item.get("props") if isinstance(item.get("props"), dict) else {}
                children = item.get("children") if isinstance(item.get("children"), list) else []

                # Extract content from old nested props format.
                content = item.get("content") or props.get("text") or props.get("prompt") or props.get("src") or ""
                if rtype in {"group", "container"} and children:
                    child_texts = [
                        str(c.get("text", c.get("content", "")))
                        for c in children if isinstance(c, dict)
                    ]
                    content = " ".join(t for t in child_texts if t) or content

                # Skip background elements.
                if rid in {"background", "bg"}:
                    continue

                role = item.get("role") or props.get("role") or (
                    "headline" if rid == "title" else
                    "subhead" if rid == "subtitle" else
                    "cta" if rid == "cta" else
                    "visual_label" if rtype == "image" else "body"
                )
                presence = item.get("presence") or props.get("presence") or (
                    "optional" if rid == "cta" else "required"
                )
                importance = item.get("priority", item.get("importance", 5))
                if isinstance(importance, str):
                    try:
                        importance = int(importance)
                    except ValueError:
                        importance = 5
                # Infer importance from role/id when not explicitly set.
                inferred = (
                    10 if role == "headline" else
                    8 if role == "subhead" else
                    7 if role == "visual_label" else 5
                )
                importance = max(importance, inferred)

                if role == "cta" and presence == "required":
                    has_cta = True
                if role == "visual_label" or rtype == "image":
                    visual_subjects.append({
                        "id": rid, "role": "illustration", "description": content or rid,
                        "presence": presence, "source": "inferred", "avoid": [],
                    })
                else:
                    messages.append({
                        "id": rid, "role": role, "content": content or rid,
                        "importance": importance, "presence": presence,
                        "source": "inferred", "editable": True,
                    })

            return {
                "poster_intent": {
                    "poster_type": "custom",
                    "communication_mode": "inform",
                    "primary_goal": payload.get("poster_goal", "Generate a poster"),
                    "target_audience": payload.get("target_audience"),
                    "tone": [],
                },
                "content_strategy": {
                    "headline_policy": "literal",
                    "information_density": "medium",
                    "cta_policy": "required" if has_cta else "omit",
                    "image_policy": "optional",
                    "inference_policy": "do_not_invent_specific_facts",
                },
                "messages": messages,
                "visual_subjects": visual_subjects,
                "must_not_do": [
                    "不要凭空编造精确日期地点",
                    "不要默认加入报名按钮",
                ],
            }

        # Ensure required top-level keys exist.
        if "poster_intent" not in payload:
            payload["poster_intent"] = {
                "poster_type": "custom",
                "communication_mode": "inform",
                "primary_goal": payload.get("poster_goal", payload.get("primary_goal", "Generate a poster")),
            }
        if "content_strategy" not in payload:
            payload["content_strategy"] = {}
        if "messages" not in payload:
            payload["messages"] = []
        if "visual_subjects" not in payload:
            payload["visual_subjects"] = []
        if "must_not_do" not in payload:
            payload["must_not_do"] = [
                "不要凭空编造精确日期地点",
                "不要默认加入报名按钮",
            ]
        return payload

    # ── validation ──

    def _validate_brief(self, brief: PosterBriefV2) -> None:
        """Ensure the PosterBriefV2 has at least one required message."""
        required_msgs = [
            m for m in brief.messages
            if m.presence == "required" and m.importance >= 8
        ]
        if not required_msgs:
            raise SchemaParseError(
                "PosterBriefV2 must contain at least one required message with importance >= 8"
            )

    # ── deterministic fallback ──

    def _run_rules(self, state: GraphState) -> ContentPlan:
        prompt = state.user_prompt.strip()
        normalized = re.sub(r"\s+", " ", prompt)
        title = self._make_title(normalized)
        reference_hint = ""
        if state.reference_images:
            reference_hint = "；参考图线索：" + "；".join(img.description for img in state.reference_images[:2])

        elements = [
            ElementContent(id="title", type=ElementType.text, content=title, priority=10, role="headline", presence="required"),
            ElementContent(
                id="subtitle",
                type=ElementType.text,
                content=self._make_subtitle(normalized),
                priority=8,
                role="subhead",
                presence="recommended",
            ),
            ElementContent(
                id="main_visual",
                type=ElementType.image,
                content=f"{normalized} 的核心视觉，干净背景，适合中文海报排版{reference_hint}",
                priority=7,
                alt="poster key visual",
                role="visual_label",
                presence="recommended",
            ),
            ElementContent(
                id="info",
                type=ElementType.text,
                content=self._make_info(normalized),
                priority=6,
                role="body",
                presence="recommended",
            ),
        ]

        if self._has_cta_intent(normalized):
            cta_text = self._make_cta(normalized)
            elements.append(
                ElementContent(
                    id="cta",
                    type=ElementType.text,
                    content=cta_text,
                    priority=5,
                    role="cta",
                    presence="required",
                )
            )

        return ContentPlan(elements=elements, poster_goal=f"为「{normalized}」生成一张清晰、有层级的宣传海报")

    def _has_cta_intent(self, prompt: str) -> bool:
        cta_keywords = [
            "报名", "购买", "预约", "扫码", "立即", "了解更多",
            "注册", "订阅", "加入", "联系", "咨询", "抢购",
            "下单", "参与", "申请", "登记", "关注",
        ]
        return any(kw in prompt for kw in cta_keywords)

    def _make_cta(self, prompt: str) -> str:
        if any(kw in prompt for kw in ["报名", "注册", "申请", "登记"]):
            return "立即报名"
        if any(kw in prompt for kw in ["购买", "抢购", "下单"]):
            return "立即购买"
        if any(kw in prompt for kw in ["预约"]):
            return "立即预约"
        if any(kw in prompt for kw in ["扫码"]):
            return "扫码参与"
        if any(kw in prompt for kw in ["订阅", "关注"]):
            return "立即关注"
        if any(kw in prompt for kw in ["联系", "咨询"]):
            return "联系我们"
        return "了解更多"

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
