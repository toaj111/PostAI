"""ContentExpander - genre-aware poster content expansion.

This agent runs after ContentExtractor. The extractor preserves user intent;
the expander asks what the poster type normally needs and adds editable
placeholder slots for missing conventional information.
"""

from __future__ import annotations

import json
from copy import deepcopy

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.llm_client import StructuredLLMClient
from app.schemas.agents import (
    ContentExpansionPlan,
    PosterBriefV2,
    PosterMessage,
    VisualSubject,
    poster_brief_to_content_plan,
)
from app.schemas.state import GraphState


class ContentExpander:
    """Expand sparse user briefs into richer poster information architecture."""

    def __init__(self, llm_client: StructuredLLMClient | None = None) -> None:
        settings = get_settings()
        self.allow_model_fallback = settings.allow_model_fallback
        self.llm_client = llm_client or StructuredLLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            response_format=settings.llm_response_format,
        )

    async def run(self, state: GraphState) -> ContentExpansionPlan:
        if state.poster_brief is None:
            raise ValueError("poster_brief is required before content expansion")

        try:
            plan = await self._run_llm(state)
        except (LLMCallError, SchemaParseError) as exc:
            if self._configured_for_llm() and not self.allow_model_fallback:
                raise
            if self._configured_for_llm():
                state.warnings.append(f"ContentExpander LLM fallback: {exc}")
            plan = self._run_rules(state)

        state.content_expansion = plan
        state.poster_brief = self.apply_expansion(state.poster_brief, plan)
        state.content_plan = poster_brief_to_content_plan(state.poster_brief)
        return plan

    async def _run_llm(self, state: GraphState) -> ContentExpansionPlan:
        if not self._configured_for_llm():
            raise LLMCallError("LLM provider is not configured")
        messages = self._build_messages(state)
        return await self.llm_client.parse(messages=messages, response_model=ContentExpansionPlan)

    def _build_messages(self, state: GraphState) -> list[dict[str, str]]:
        brief_json = state.poster_brief.model_dump(mode="json") if state.poster_brief else None
        system = (
            "You are a senior poster editor doing genre-aware content expansion.\n"
            "Return only JSON matching ContentExpansionPlan.\n\n"
            "Before style or layout, ask yourself: for this poster type, what "
            "important poster information did the user not mention? Add common "
            "slots only when they improve the poster's information architecture.\n\n"
            "Do NOT fabricate precise facts. Never invent real dates, venues, "
            "artist names, prices, URLs, sponsors, company facts, phone numbers, "
            "or QR destinations. Use editable placeholders such as DATE TBD, "
            "VENUE TBD, LINEUP TBA, TICKETS INFO, SCHEDULE TBA, or Chinese "
            "equivalents when the fact is missing.\n\n"
            "Mark generated content source='placeholder' for missing facts and "
            "source='inferred' for generic section labels or mood copy. Use "
            "presence='recommended' for genre-critical placeholders, and "
            "presence='optional' for nice-to-have supporting details. Only use "
            "presence='required' when the original brief already requires that "
            "information or the poster cannot function without it.\n\n"
            "For music/event posters, consider date, venue, lineup, stages, "
            "schedule, ticket info, age/access note, CTA, and stage/crowd/lighting "
            "visual motifs. For recruitment, consider role, location, salary, "
            "benefits, requirements, CTA/QR. For product, consider product name, "
            "value proposition, feature slots, launch/availability, CTA. For "
            "informational posters, consider schedule blocks, location, organizer, "
            "and section labels.\n\n"
            "If the user explicitly asks for minimal, type-only, abstract-only, "
            "or no extra information, keep density sparse and add little or no "
            "content expansion."
        )
        user = (
            f"User prompt: {state.user_prompt}\n"
            f"Canvas: {state.canvas.width}x{state.canvas.height}px\n\n"
            f"Existing PosterBriefV2:\n{json.dumps(brief_json, ensure_ascii=False)}\n\n"
            "Create a ContentExpansionPlan. The output should explain the "
            "self-questions and add only useful inferred/placeholder messages "
            "and visual subjects that are not already covered by the brief."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def apply_expansion(
        self,
        brief: PosterBriefV2,
        plan: ContentExpansionPlan,
    ) -> PosterBriefV2:
        expanded = brief.model_copy(deep=True)

        existing_message_keys = {
            self._message_key(message)
            for message in expanded.messages
        }
        existing_ids = {message.id for message in expanded.messages}
        for message in plan.inferred_messages:
            if message.presence == "omit":
                continue
            if message.id in existing_ids or self._message_key(message) in existing_message_keys:
                continue
            expanded.messages.append(deepcopy(message))
            existing_ids.add(message.id)
            existing_message_keys.add(self._message_key(message))

        existing_visual_keys = {
            self._visual_key(subject)
            for subject in expanded.visual_subjects
        }
        existing_visual_ids = {subject.id for subject in expanded.visual_subjects}
        for subject in plan.inferred_visual_subjects:
            if subject.presence == "omit" or subject.role == "none":
                continue
            if subject.id in existing_visual_ids or self._visual_key(subject) in existing_visual_keys:
                continue
            expanded.visual_subjects.append(deepcopy(subject))
            existing_visual_ids.add(subject.id)
            existing_visual_keys.add(self._visual_key(subject))

        if plan.density_recommendation in {"sparse", "medium", "dense"}:
            if expanded.content_strategy.information_density != "sparse":
                expanded.content_strategy.information_density = plan.density_recommendation
        if self._has_cta(expanded) and expanded.content_strategy.cta_policy == "omit":
            expanded.content_strategy.cta_policy = "optional"
        expanded.content_strategy.inference_policy = "allow_genre_placeholders_no_specific_facts"

        for rule in plan.must_not_invent:
            if rule not in expanded.must_not_do:
                expanded.must_not_do.append(rule)

        return expanded

    def _run_rules(self, state: GraphState) -> ContentExpansionPlan:
        brief = state.poster_brief
        if brief is None:
            raise ValueError("poster_brief is required before content expansion")

        if self._wants_sparse(state.user_prompt, brief):
            return ContentExpansionPlan(
                poster_type=brief.poster_intent.poster_type,
                density_recommendation="sparse",
                self_questions=[
                    "Did the user explicitly ask for minimal, type-only, or abstract-only output?",
                    "Would extra factual slots weaken the requested minimal poster?",
                ],
                assumptions=["The prompt asks for restrained content; keep expansion minimal."],
                must_not_invent=self._default_must_not_invent(),
            )

        poster_type = self._infer_expansion_type(state)
        if poster_type == "music_event":
            return self._music_event_plan(state)
        if poster_type == "recruitment":
            return self._recruitment_plan(state)
        if poster_type == "product":
            return self._product_plan(state)
        if poster_type == "informational":
            return self._informational_plan(state)
        if poster_type == "event":
            return self._event_plan(state)
        return self._generic_plan(state)

    def _music_event_plan(self, state: GraphState) -> ContentExpansionPlan:
        return ContentExpansionPlan(
            poster_type="music_event",
            density_recommendation="dense",
            self_questions=[
                "What does a music festival poster usually need beyond the title?",
                "Which missing facts should be shown as editable placeholders instead of omitted?",
                "What music-specific visual motifs can give the layout more substance?",
            ],
            assumptions=[
                "The prompt asks for a music festival/event poster, so date, venue, lineup, stages, and ticket information are expected slots.",
                "Specific artists, exact time, and exact location are unknown, so they must remain placeholders.",
            ],
            inferred_messages=[
                self._msg("date_tbd", "date", "DATE TBD", 8, "recommended", "placeholder", "Music festival posters normally show the date."),
                self._msg("venue_tbd", "venue", "VENUE TBD", 8, "recommended", "placeholder", "Music festival posters normally show the venue."),
                self._msg("lineup_tba", "body", "LINEUP TBA", 8, "recommended", "placeholder", "Lineup is central to a festival poster but artist names are unknown."),
                self._msg("stage_program", "meta", "MAIN STAGE / BASS STAGE / DANCE STAGE", 6, "recommended", "inferred", "Generic stage labels add event structure without inventing performers."),
                self._msg("time_window", "meta", "18:00 - LATE", 5, "optional", "placeholder", "A time slot is useful but not user-provided."),
                self._msg("ticket_info", "cta", "TICKETS INFO", 6, "optional", "placeholder", "Ticket information is useful for event posters but should not become required unless requested."),
            ],
            inferred_visual_subjects=[
                VisualSubject(
                    id="stage-light-system",
                    role="symbol",
                    description="Layered stage-light beams, waveform rhythm, speaker silhouettes, and crowd-energy texture around the headline",
                    presence="recommended",
                    source="inferred",
                    avoid=["real artist portraits", "specific sponsor logos", "fake venue branding"],
                ),
                VisualSubject(
                    id="lineup-grid-texture",
                    role="pattern",
                    description="Festival lineup grid or ticket-stub pattern used as a secondary texture field",
                    presence="recommended",
                    source="inferred",
                    avoid=["invented performer names"],
                ),
            ],
            must_not_invent=self._default_must_not_invent(),
        )

    def _event_plan(self, state: GraphState) -> ContentExpansionPlan:
        return ContentExpansionPlan(
            poster_type="event",
            density_recommendation="medium",
            self_questions=[
                "What event details would make this poster usable?",
                "Which missing details should remain TBD placeholders?",
            ],
            assumptions=["The prompt reads as an event announcement with missing logistics."],
            inferred_messages=[
                self._msg("date_tbd", "date", "DATE TBD", 8, "recommended", "placeholder", "Event posters normally show date."),
                self._msg("venue_tbd", "venue", "VENUE TBD", 8, "recommended", "placeholder", "Event posters normally show venue."),
                self._msg("program_tba", "body", "PROGRAM TBA", 6, "optional", "placeholder", "Program details are unknown."),
            ],
            inferred_visual_subjects=[
                VisualSubject(
                    id="event-structure-system",
                    role="shape",
                    description="Directional event graphics with schedule strip, registration marks, and a clear metadata zone",
                    presence="recommended",
                    source="inferred",
                )
            ],
            must_not_invent=self._default_must_not_invent(),
        )

    def _recruitment_plan(self, state: GraphState) -> ContentExpansionPlan:
        return ContentExpansionPlan(
            poster_type="recruitment",
            density_recommendation="dense",
            self_questions=[
                "Which recruitment facts does the poster normally need?",
                "Which details are missing and should remain editable placeholders?",
            ],
            assumptions=["Recruitment posters need role, benefits, requirements, and an application path."],
            inferred_messages=[
                self._msg("role_focus", "body", "岗位方向：职位待补充", 8, "recommended", "placeholder", "Role focus is useful but exact role may be missing."),
                self._msg("salary_placeholder", "body", "薪资待遇：面议 / 待补充", 8, "recommended", "placeholder", "Salary should not be invented."),
                self._msg("benefits_placeholder", "body", "福利亮点：待补充", 7, "recommended", "placeholder", "Benefits are expected but unknown."),
                self._msg("requirements_placeholder", "body", "岗位要求：待补充", 7, "recommended", "placeholder", "Requirements are expected but unknown."),
                self._msg("application_info", "cta", "投递 / 报名方式待补充", 7, "optional", "placeholder", "Application path is useful but unknown."),
            ],
            inferred_visual_subjects=[
                VisualSubject(
                    id="career-path-system",
                    role="illustration",
                    description="Campus/career path motif with modular info cards and a QR/action zone",
                    presence="recommended",
                    source="inferred",
                )
            ],
            must_not_invent=self._default_must_not_invent(),
        )

    def _product_plan(self, state: GraphState) -> ContentExpansionPlan:
        return ContentExpansionPlan(
            poster_type="product",
            density_recommendation="medium",
            self_questions=[
                "What does a product poster need beyond the product theme?",
                "Which feature and availability facts are missing?",
            ],
            assumptions=["Product posters benefit from value proposition, feature slots, and availability/CTA placeholders."],
            inferred_messages=[
                self._msg("value_prop", "subhead", "核心卖点待补充", 8, "recommended", "placeholder", "A product poster needs a value proposition."),
                self._msg("feature_slots", "body", "FEATURES TBA", 7, "recommended", "placeholder", "Feature details are unknown."),
                self._msg("availability", "meta", "AVAILABLE TBD", 5, "optional", "placeholder", "Availability is useful but unknown."),
            ],
            inferred_visual_subjects=[
                VisualSubject(
                    id="product-showcase-system",
                    role="illustration",
                    description="Studio product plinth, feature callout rails, and premium surface texture",
                    presence="recommended",
                    source="inferred",
                )
            ],
            must_not_invent=self._default_must_not_invent(),
        )

    def _informational_plan(self, state: GraphState) -> ContentExpansionPlan:
        return ContentExpansionPlan(
            poster_type="informational",
            density_recommendation="dense",
            self_questions=[
                "What section labels and schedule blocks would make the poster scannable?",
                "Which facts should stay as placeholders?",
            ],
            assumptions=["Information posters need structured sections even when the prompt is short."],
            inferred_messages=[
                self._msg("schedule_block", "body", "SCHEDULE TBA", 8, "recommended", "placeholder", "Schedule information is expected but unknown."),
                self._msg("location_tbd", "venue", "LOCATION TBD", 7, "recommended", "placeholder", "Location is expected but unknown."),
                self._msg("organizer_tbd", "meta", "ORGANIZER TBD", 5, "optional", "placeholder", "Organizer is optional and unknown."),
            ],
            inferred_visual_subjects=[
                VisualSubject(
                    id="information-grid-system",
                    role="frame",
                    description="Modular grid, numbered sections, and timeline rails for dense readable information",
                    presence="recommended",
                    source="inferred",
                )
            ],
            must_not_invent=self._default_must_not_invent(),
        )

    def _generic_plan(self, state: GraphState) -> ContentExpansionPlan:
        return ContentExpansionPlan(
            poster_type=state.poster_brief.poster_intent.poster_type if state.poster_brief else "custom",
            density_recommendation="medium",
            self_questions=[
                "Does this prompt imply a conventional poster type with missing details?",
                "Would placeholders add clarity without fabricating facts?",
            ],
            assumptions=["No strong genre expansion rule matched; keep additions light."],
            inferred_messages=[
                self._msg("supporting_detail", "meta", "DETAILS TBD", 5, "optional", "placeholder", "A light metadata slot can help sparse prompts without forcing facts."),
            ],
            inferred_visual_subjects=[],
            must_not_invent=self._default_must_not_invent(),
        )

    def _infer_expansion_type(self, state: GraphState) -> str:
        brief = state.poster_brief
        prompt = state.user_prompt.lower()
        poster_type = brief.poster_intent.poster_type.lower() if brief else "custom"
        text = f"{state.user_prompt} {poster_type} {' '.join(brief.poster_intent.tone if brief else [])}".lower()

        if any(token in text for token in ("音乐", "音乐节", "演出", "live", "festival", "concert", "jazz", "dj")):
            return "music_event"
        if any(token in text for token in ("招聘", "岗位", "校园", "recruit", "career", "hiring")) or poster_type == "recruitment":
            return "recruitment"
        if any(token in text for token in ("产品", "发布", "新品", "product", "launch")) or poster_type == "product":
            return "product"
        if any(token in text for token in ("日程", "讲座", "课程", "论坛", "schedule", "agenda")) or poster_type == "informational":
            return "informational"
        if poster_type in {"event", "campaign"} or any(token in text for token in ("活动", "会议", "展会", "event", "conference")):
            return "event"
        return poster_type

    def _wants_sparse(self, prompt: str, brief: PosterBriefV2) -> bool:
        text = f"{prompt} {' '.join(brief.poster_intent.tone)}".lower()
        sparse_tokens = (
            "极简",
            "纯文字",
            "只用文字",
            "不要图",
            "不要额外信息",
            "抽象",
            "minimal",
            "type-only",
            "only text",
            "abstract-only",
            "no extra",
        )
        if any(token in text for token in sparse_tokens):
            return True
        return (
            brief.content_strategy.information_density == "sparse"
            and brief.content_strategy.image_policy == "omit"
        )

    def _msg(
        self,
        item_id: str,
        role: str,
        content: str,
        importance: int,
        presence: str,
        source: str,
        notes: str,
    ) -> PosterMessage:
        return PosterMessage(
            id=item_id,
            role=role,
            content=content,
            importance=importance,
            presence=presence,
            source=source,
            editable=True,
            notes=notes,
        )

    def _has_cta(self, brief: PosterBriefV2) -> bool:
        return any(message.role == "cta" and message.presence != "omit" for message in brief.messages)

    def _message_key(self, message: PosterMessage) -> tuple[str, str]:
        return (message.role.lower(), message.content.strip().lower())

    def _visual_key(self, subject: VisualSubject) -> tuple[str, str]:
        return (subject.role.lower(), subject.description.strip().lower())

    def _default_must_not_invent(self) -> list[str]:
        return [
            "Do not invent exact dates, venues, prices, URLs, sponsors, or real names.",
            "Placeholder slots must stay editable and visibly provisional.",
        ]

    def _configured_for_llm(self) -> bool:
        return bool(
            self.llm_client.api_key
            and self.llm_client.base_url
            and not self.llm_client.model.startswith("mock-")
        )
