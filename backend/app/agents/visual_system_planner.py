"""VisualSystemPlanner - executable visual layer planning.

This agent sits between ArtDirectionV2 and raw HTML generation. It turns the
brief/style intent into a concrete layer stack so the layout model has specific
poster devices to implement instead of a vague instruction to "add richness".
"""

from __future__ import annotations

import json

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.llm_client import StructuredLLMClient
from app.schemas.agents import VisualLayerSpec, VisualSystemPlan
from app.schemas.state import GraphState


class VisualSystemPlanner:
    """Plan the poster's visible layer system before HTML/CSS generation."""

    def __init__(self, llm_client: StructuredLLMClient | None = None) -> None:
        settings = get_settings()
        self.allow_model_fallback = settings.allow_model_fallback
        self.llm_client = llm_client or StructuredLLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            response_format=settings.llm_response_format,
        )

    async def run(self, state: GraphState) -> VisualSystemPlan:
        try:
            plan = await self._run_llm(state)
            self._validate_plan(plan, state)
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_llm() and not self.allow_model_fallback:
                raise
            if self._configured_for_llm():
                state.warnings.append(f"VisualSystemPlanner LLM fallback: {exc}")
            plan = self._run_fallback(state)

        state.visual_system = plan
        return plan

    async def _run_llm(self, state: GraphState) -> VisualSystemPlan:
        if not self._configured_for_llm():
            raise LLMCallError("LLM provider is not configured")
        messages = self._build_messages(state)
        return await self._parse_visual_system(messages)

    def _build_messages(self, state: GraphState) -> list[dict[str, str]]:
        brief_json = state.poster_brief.model_dump(mode="json") if state.poster_brief else None
        ad_json = state.art_direction.model_dump(mode="json") if state.art_direction else None
        feedback_text = self._build_feedback_context(state)

        system = (
            "You are a senior poster visual-systems designer.\n"
            "Return only JSON matching VisualSystemPlan.\n\n"
            "Your job is to translate PosterBriefV2 and ArtDirectionV2 into an "
            "executable poster layer stack for an HTML/CSS production artist.\n"
            "Think in visible layers: base field, texture/noise/grid, cropped type, "
            "directional geometry, focal symbol/image, frame/rules, metadata strip, "
            "foreground text hierarchy, CTA/QR when required.\n\n"
            "Do NOT design a landing page, dashboard, web card, button-first hero, "
            "or plain centered template.\n"
            "Do NOT invent factual dates, venues, prices, URLs, sponsors, or logos.\n"
            "Do NOT ask for a CTA/QR unless the brief requires it.\n\n"
            "For non-minimal posters, target 6-9 purposeful layers and set "
            "layer_count_target accordingly. For type-only or extreme minimal "
            "posters, target 3-5 layers, but still include deliberate type, spacing, "
            "rules, and material treatment.\n\n"
            "Every layer must have a stable id suitable for an HTML id attribute. "
            "Use concrete CSS approaches such as repeating-linear-gradient, inline "
            "svg, clip-path, mix-blend-mode, mask, border rules, grid, oversized "
            "cropped text, or object-fit image crop. Avoid generic small icons.\n"
            "Set required_html_ids to the ids of all layers whose presence is "
            "required, plus important recommended layers that define the poster idea."
        )

        user_parts = [
            f"User prompt: {state.user_prompt}",
            f"Canvas: {state.canvas.width}x{state.canvas.height}px",
            "",
            f"PosterBriefV2:\n{json.dumps(brief_json, ensure_ascii=False) if brief_json else 'none'}",
            "",
            f"ArtDirectionV2:\n{json.dumps(ad_json, ensure_ascii=False) if ad_json else 'none'}",
        ]

        if feedback_text:
            user_parts.extend(["", feedback_text])

        if state.reference_images:
            refs = [
                f"{index}. {image.url} | {image.description}"
                for index, image in enumerate(state.reference_images, start=1)
            ]
            user_parts.extend(["", "Reference images:\n" + "\n".join(refs)])

        user_parts.append("")
        user_parts.append(
            "Create the VisualSystemPlan now. Make it specific enough that the "
            "HTML planner cannot collapse the poster into a sparse centered layout."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    async def _parse_visual_system(self, messages: list[dict[str, str]]) -> VisualSystemPlan:
        try:
            return await self.llm_client.parse(messages=messages, response_model=VisualSystemPlan)
        except SchemaParseError:
            if not hasattr(self.llm_client, "_chat_completion"):
                raise
            content = await self.llm_client._chat_completion(
                messages=messages,
                response_model=VisualSystemPlan,
            )
            payload = json.loads(content)
            normalized = self._normalize_plan_payload(payload)
            return self.llm_client.validate_payload(normalized, VisualSystemPlan)

    def _normalize_plan_payload(self, payload: dict) -> dict:
        if "layers" not in payload:
            payload["layers"] = []
        if "required_html_ids" not in payload:
            payload["required_html_ids"] = [
                str(layer.get("id", ""))
                for layer in payload.get("layers", [])
                if isinstance(layer, dict) and layer.get("presence") == "required" and layer.get("id")
            ]
        if "composition_archetype" not in payload:
            payload["composition_archetype"] = payload.get("composition", "centered_iconic")
        if "density" not in payload:
            payload["density"] = "medium"
        if "layer_count_target" not in payload:
            payload["layer_count_target"] = max(4, min(9, len(payload.get("layers", [])) or 6))
        if "constraints" not in payload:
            payload["constraints"] = []
        return payload

    def _validate_plan(self, plan: VisualSystemPlan, state: GraphState) -> None:
        if not plan.layers:
            raise SchemaParseError("VisualSystemPlan must include at least one layer")

        ids = [layer.id for layer in plan.layers if layer.presence != "omit"]
        if len(ids) != len(set(ids)):
            raise SchemaParseError("VisualSystemPlan layer ids must be unique")

        if not self._allows_sparse_system(state) and plan.layer_count_target < 5:
            raise SchemaParseError("Non-minimal posters need a layer_count_target >= 5")

    def _run_fallback(self, state: GraphState) -> VisualSystemPlan:
        brief = state.poster_brief
        ad = state.art_direction
        sparse = self._allows_sparse_system(state)

        if ad is not None:
            composition = ad.poster_language.composition_family
            density = ad.poster_language.visual_density
            type_notes = (
                f"{ad.typography.headline_style}, scale_contrast={ad.typography.scale_contrast}"
            )
            image_strategy = f"{ad.imagery.treatment} / {ad.imagery.background_strategy}"
        else:
            composition = "centered_iconic"
            density = "sparse" if sparse else "medium"
            type_notes = "bold poster typography with clear hierarchy"
            image_strategy = "abstract_geometry / pattern"

        visual_subjects = [
            subject for subject in (brief.visual_subjects if brief else [])
            if subject.presence != "omit" and subject.role != "none"
        ]
        cta_required = bool(
            brief and (
                brief.content_strategy.cta_policy == "required"
                or any(message.role == "cta" and message.presence == "required" for message in brief.messages)
            )
        )

        target = 4 if sparse else (8 if density == "dense" else 7)
        layers: list[VisualLayerSpec] = [
            VisualLayerSpec(
                id="base-field",
                role="background",
                description="Full-canvas colour field based on the art direction palette",
                purpose="Establish poster mood and edge-to-edge print surface",
                placement="full canvas",
                scale="full_bleed",
                css_approach="solid or layered linear/radial gradients",
                priority=10,
                presence="required",
            ),
            VisualLayerSpec(
                id="headline-system",
                role="typography",
                description="Primary headline with oversized poster-scale type treatment",
                purpose="Create the main communication hierarchy",
                placement="dominant foreground zone with intentional crop or alignment",
                scale="large_to_oversized",
                css_approach="font-weight, line-height, grid alignment, possible text-shadow or clipping",
                priority=10,
                presence="required",
            ),
        ]

        if not sparse:
            layers.insert(
                1,
                VisualLayerSpec(
                    id="texture-field",
                    role="texture",
                    description="Subtle grid, grain, paper, or scan-line texture",
                    purpose="Prevent empty flatness and give the poster a material surface",
                    placement="full canvas, behind content",
                    scale="full_bleed",
                    css_approach="repeating-linear-gradient, radial-gradient speckles, mix-blend-mode",
                    priority=8,
                    presence="required",
                ),
            )
            visual_description = (
                visual_subjects[0].description
                if visual_subjects
                else f"theme-specific abstract focal device using {image_strategy}"
            )
            layers.append(
                VisualLayerSpec(
                    id="key-visual-system",
                    role=visual_subjects[0].role if visual_subjects else "symbol",
                    description=visual_description,
                    purpose="Provide a concrete focal device tied to the poster topic",
                    placement="large cropped zone, overlapping background and text rhythm",
                    scale="25%-45% canvas, intentionally cropped",
                    css_approach="inline svg, clip-path, layered shapes, object-fit image crop",
                    priority=8,
                    presence="required" if visual_subjects else "recommended",
                )
            )
            layers.append(
                VisualLayerSpec(
                    id="kinetic-shape",
                    role="shape",
                    description="Directional shape or band that moves the eye through the canvas",
                    purpose="Add composition energy and connect text to the focal visual",
                    placement="diagonal or offset across the middle third",
                    scale="medium_to_large",
                    css_approach="clip-path polygon, skewed block, border, transform",
                    priority=7,
                    presence="recommended",
                )
            )

        layers.append(
            VisualLayerSpec(
                id="rule-frame",
                role="frame",
                description="Frame, rules, or registration marks used as a poster structure",
                purpose="Make the composition feel designed rather than floating",
                placement="near edges and/or key grid lines",
                scale="thin structural accents",
                css_approach="absolute borders, pseudo-elements, rule lines",
                priority=6,
                presence="recommended",
            )
        )
        layers.append(
            VisualLayerSpec(
                id="metadata-strip",
                role="metadata",
                description="Small label/strip/index text that supports poster identity without inventing facts",
                purpose="Add editorial detail and scale contrast",
                placement="edge, corner, or vertical side rail",
                scale="small",
                css_approach="writing-mode, grid row, tiny uppercase label",
                priority=5,
                presence="recommended" if not sparse else "optional",
            )
        )

        if cta_required:
            layers.append(
                VisualLayerSpec(
                    id="action-system",
                    role="cta",
                    description="Required registration/action area from the brief",
                    purpose="Make the conversion action visible without turning the poster into a web UI",
                    placement="lower third or side rail, away from headline overlap",
                    scale="medium",
                    css_approach="high-contrast block, QR placeholder if prompt asks for scan code",
                    priority=7,
                    presence="required",
                )
            )

        required_html_ids = [
            layer.id
            for layer in layers
            if layer.presence == "required" or (not sparse and layer.priority >= 7)
        ]

        return VisualSystemPlan(
            composition_archetype=composition,
            density="sparse" if sparse else density,
            focal_strategy=(
                "Use the headline and key visual as competing anchors with a directional rhythm"
                if not sparse
                else "Use scale, alignment, and negative space as the primary visual idea"
            ),
            layer_count_target=target,
            required_html_ids=required_html_ids,
            layers=layers,
            typography_treatment=type_notes,
            rhythm_notes="Vary scale, opacity, crop, and alignment; avoid one centered stack.",
            constraints=[
                "Do not invent factual dates, venues, prices, sponsors, or URLs.",
                "Do not use generic UI cards or rounded landing-page buttons.",
                "Do not collapse the plan into one small icon plus centered text.",
            ],
        )

    def _build_feedback_context(self, state: GraphState) -> str:
        if not state.feedback_history:
            return ""

        latest = state.feedback_history[-1]
        parts: list[str] = ["Previous visual critique:"]
        if latest.vision_description:
            parts.append(f"What was visible: {latest.vision_description}")
        if latest.issues:
            parts.append(f"Issues: {json.dumps(latest.issues, ensure_ascii=False)}")
        if latest.suggestions:
            parts.append(f"Suggestions: {json.dumps(latest.suggestions, ensure_ascii=False)}")
        if latest.structured_issues:
            parts.append(
                "Structured issues: "
                + json.dumps(
                    [issue.model_dump(mode="json") for issue in latest.structured_issues],
                    ensure_ascii=False,
                )
            )
        parts.append("Revise the visual layer plan to address this before HTML generation.")
        return "\n".join(parts)

    def _allows_sparse_system(self, state: GraphState) -> bool:
        brief = state.poster_brief
        ad = state.art_direction
        prompt = state.user_prompt.lower()
        sparse_tokens = (
            "\u6781\u7b80",
            "\u7eaf\u6587\u5b57",
            "\u53ea\u7528\u6587\u5b57",
            "\u4e0d\u8981\u56fe",
            "minimal",
            "type-only",
            "typographic",
            "only text",
            "no image",
        )
        if any(token in prompt for token in sparse_tokens):
            return True
        if brief is not None:
            poster_type = brief.poster_intent.poster_type.lower()
            tones = [tone.lower() for tone in brief.poster_intent.tone]
            if poster_type in {"typographic", "artistic"} and any(tone in {"minimal", "sparse"} for tone in tones):
                return True
            if brief.content_strategy.image_policy == "omit" and brief.content_strategy.information_density == "sparse":
                return True
        if ad is not None:
            pl = ad.poster_language
            return (
                pl.visual_density == "sparse"
                and pl.negative_space == "generous"
                and pl.composition_family in {"minimal", "typographic"}
            )
        return False

    def _configured_for_llm(self) -> bool:
        return bool(
            self.llm_client.api_key
            and self.llm_client.base_url
            and not self.llm_client.model.startswith("mock-")
        )
