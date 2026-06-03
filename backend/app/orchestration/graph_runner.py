"""Graph runner — Phase 2 (HTML/CSS pipeline).

Orchestrates the multi-agent poster-generation pipeline.
The layout planner now produces an HTML document; the renderer is an
``HTMLPainter`` that captures it via headless Chromium.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.agents.content_expander import ContentExpander
from app.agents.content_extractor import ContentExtractor
from app.agents.layout_planner import SpatialLayoutPlanner
from app.agents.style_director import StyleDirector
from app.agents.visual_system_planner import VisualSystemPlanner
from app.agents.vlm_critic import HeuristicVLMCritic
from app.core.config import get_settings
from app.core.events import SSEEvent, event
from app.orchestration.retry import retry_async
from app.orchestration.router import RouteAction, route_after_critique
from app.render.asset_store import AssetStore
from app.render.html_painter import apply_canvas_guard
from app.schemas.api import GenerateResponse
from app.schemas.state import GraphStage, GraphState


class GraphRunner:
    """Top-level orchestrator that wires agents together.

    Default dependencies are created for tests / quick start; in production
    you can inject pre-configured agent instances.
    """

    def __init__(
        self,
        content_extractor: ContentExtractor | None = None,
        content_expander: ContentExpander | None = None,
        style_director: StyleDirector | None = None,
        visual_system_planner: VisualSystemPlanner | None = None,
        layout_planner: SpatialLayoutPlanner | None = None,
        renderer=None,
        asset_store: AssetStore | None = None,
        critic: HeuristicVLMCritic | None = None,
    ) -> None:
        from app.render.html_painter import HTMLPainter

        settings = get_settings()
        self.content_extractor = content_extractor or ContentExtractor()
        self.content_expander = content_expander or ContentExpander()
        self.style_director = style_director or StyleDirector()
        self.visual_system_planner = visual_system_planner or VisualSystemPlanner()
        self.layout_planner = layout_planner or SpatialLayoutPlanner()
        self.renderer = renderer or HTMLPainter()
        self.asset_store = asset_store or AssetStore(settings.asset_dir, settings.asset_url_path)
        self.critic = critic or HeuristicVLMCritic()

    # ── streaming (SSE) ──

    async def run_events(self, state: GraphState) -> AsyncIterator[SSEEvent]:
        try:
            yield event("job_started", {"job_id": state.job_id})
            warning_cursor = 0

            def _pop_new_warnings() -> list[str]:
                nonlocal warning_cursor
                warnings = state.warnings[warning_cursor:]
                warning_cursor = len(state.warnings)
                return warnings

            def _mark_warnings_seen() -> None:
                nonlocal warning_cursor
                warning_cursor = len(state.warnings)

            # ── Content ──
            state.stage = GraphStage.content
            yield event("agent_start", {"job_id": state.job_id, "agent": "ContentExtractor", "message": "Parsing poster content"})
            state.content_plan = await retry_async(lambda: self.content_extractor.run(state), attempts=3)
            yield event("agent_complete", {"job_id": state.job_id, "agent": "ContentExtractor", "result": state.content_plan.model_dump(mode="json")})
            for warning in _pop_new_warnings():
                yield event("warning", {"job_id": state.job_id, "message": warning})

            state.stage = GraphStage.content_expansion
            yield event("agent_start", {"job_id": state.job_id, "agent": "ContentExpander", "message": "Expanding genre-specific poster information"})
            state.content_expansion = await retry_async(lambda: self.content_expander.run(state), attempts=3)
            yield event("agent_complete", {"job_id": state.job_id, "agent": "ContentExpander", "result": state.content_expansion.model_dump(mode="json")})
            for warning in _pop_new_warnings():
                yield event("warning", {"job_id": state.job_id, "message": warning})

            # ── Style ──
            state.stage = GraphStage.style
            yield event("agent_start", {"job_id": state.job_id, "agent": "StyleDirector", "message": "Planning visual style"})
            state.style = await retry_async(lambda: self.style_director.run(state), attempts=3)
            yield event("agent_complete", {"job_id": state.job_id, "agent": "StyleDirector", "result": state.style.model_dump(mode="json")})
            for warning in _pop_new_warnings():
                yield event("warning", {"job_id": state.job_id, "message": warning})

            while state.iteration_count < state.max_iterations:
                # ── Visual System ──
                state.stage = GraphStage.visual_system
                yield event("agent_start", {"job_id": state.job_id, "agent": "VisualSystemPlanner", "message": "Planning executable visual layer system"})
                state.visual_system = await retry_async(lambda: self.visual_system_planner.run(state), attempts=3)
                yield event("agent_complete", {"job_id": state.job_id, "agent": "VisualSystemPlanner", "result": state.visual_system.model_dump(mode="json")})
                for warning in _pop_new_warnings():
                    yield event("warning", {"job_id": state.job_id, "message": warning})

                # ── Layout (HTML) ──
                state.stage = GraphStage.layout
                yield event("agent_start", {"job_id": state.job_id, "agent": "SpatialLayoutPlanner", "message": "Designing poster HTML/CSS"})
                state.layout_html = await retry_async(lambda: self.layout_planner.run(state), attempts=3)
                state.layout_html = apply_canvas_guard(
                    state.layout_html,
                    width=state.canvas.width,
                    height=state.canvas.height,
                )
                # Persist the HTML source alongside the rendered PNG.
                state.html_url = await self.asset_store.save_html(
                    state.layout_html, job_id=state.job_id, iteration=state.iteration_count
                )
                preview = state.layout_html[:200] + "..." if len(state.layout_html) > 200 else state.layout_html
                yield event("agent_complete", {"job_id": state.job_id, "agent": "SpatialLayoutPlanner", "result": {"html_preview": preview, "html_url": state.html_url}})
                for warning in _pop_new_warnings():
                    yield event("warning", {"job_id": state.job_id, "message": warning})

                # ── Render (Playwright) ──
                state.stage = GraphStage.render
                yield event("agent_start", {"job_id": state.job_id, "agent": "HTMLPainter", "message": "Rendering poster via headless browser"})
                state.render_result = await retry_async(
                    lambda: self.renderer.render(state.layout_html, width=state.canvas.width, height=state.canvas.height),
                    attempts=2,
                )
                state.render_result = await self.asset_store.save_render(
                    state.render_result,
                    job_id=state.job_id,
                    iteration=state.iteration_count,
                )
                yield event("render_preview", {"job_id": state.job_id, "iteration": state.iteration_count, **state.render_result.model_dump(mode="json")})

                # ── Critique (VLM) ──
                state.stage = GraphStage.critique
                yield event("agent_start", {"job_id": state.job_id, "agent": "VLMCritic", "message": "Reviewing visual result"})
                critique = await retry_async(lambda: self.critic.run(state), attempts=2)
                state.feedback_history.append(critique)
                yield event("critique", {
                    "job_id": state.job_id,
                    **critique.model_dump(mode="json"),
                    "vision_reasoning": state.vision_reasoning,
                })
                for warning in _pop_new_warnings():
                    yield event("warning", {"job_id": state.job_id, "message": warning})

                # ── Decision ──
                decision = route_after_critique(state)
                if decision.action == RouteAction.final:
                    if critique.score < state.target_score and not critique.passed:
                        state.warnings.append(decision.reason)
                        yield event("warning", {"job_id": state.job_id, "message": decision.reason})
                        _mark_warnings_seen()
                    break

                state.iteration_count += 1
                if decision.action == RouteAction.style:
                    state.stage = GraphStage.style
                    yield event("agent_start", {"job_id": state.job_id, "agent": "StyleDirector", "message": decision.reason})
                    state.style = await retry_async(lambda: self.style_director.run(state), attempts=3)
                    yield event("agent_complete", {"job_id": state.job_id, "agent": "StyleDirector", "result": state.style.model_dump(mode="json")})
                    for warning in _pop_new_warnings():
                        yield event("warning", {"job_id": state.job_id, "message": warning})
                elif decision.action == RouteAction.content:
                    state.stage = GraphStage.content
                    yield event("agent_start", {"job_id": state.job_id, "agent": "ContentExtractor", "message": decision.reason})
                    state.content_plan = await retry_async(lambda: self.content_extractor.run(state), attempts=3)
                    yield event("agent_complete", {"job_id": state.job_id, "agent": "ContentExtractor", "result": state.content_plan.model_dump(mode="json")})
                    for warning in _pop_new_warnings():
                        yield event("warning", {"job_id": state.job_id, "message": warning})

                    state.stage = GraphStage.content_expansion
                    yield event("agent_start", {"job_id": state.job_id, "agent": "ContentExpander", "message": "Re-expanding poster information after content feedback"})
                    state.content_expansion = await retry_async(lambda: self.content_expander.run(state), attempts=3)
                    yield event("agent_complete", {"job_id": state.job_id, "agent": "ContentExpander", "result": state.content_expansion.model_dump(mode="json")})
                    for warning in _pop_new_warnings():
                        yield event("warning", {"job_id": state.job_id, "message": warning})

            # ── Finalise ──
            state.stage = GraphStage.final
            response = self._finalize_response(state)
            yield event("final_output", response.model_dump(mode="json"))
            yield event("job_finished", {"job_id": state.job_id, "stage": state.stage.value})

        except Exception as exc:
            failed_stage = state.stage.value
            state.stage = GraphStage.error
            state.error = str(exc)
            yield event("error", {
                "job_id": state.job_id,
                "stage": failed_stage,
                "message": str(exc),
                "recoverable": False,
            })

    # ── synchronous convenience ──

    async def run(self, state: GraphState) -> GenerateResponse:
        final: GenerateResponse | None = None
        async for sse_event in self.run_events(state):
            if sse_event.event == "final_output":
                final = GenerateResponse.model_validate(sse_event.data)
            if sse_event.event == "error":
                raise RuntimeError(str(sse_event.data.get("message", "generation failed")))
        if final is None:
            raise RuntimeError("generation did not produce final output")
        return final

    def build_response(self, state: GraphState) -> GenerateResponse:
        latest = state.feedback_history[-1] if state.feedback_history else None
        return GenerateResponse(
            job_id=state.job_id,
            final_image=state.render_result.image_base64 if state.render_result else None,
            image_url=state.render_result.image_url if state.render_result else None,
            score=latest.score if latest else None,
            warnings=state.warnings,
            content_plan=state.content_plan,
            poster_brief=state.poster_brief,
            content_expansion=state.content_expansion,
            art_direction=state.art_direction,
            visual_system=state.visual_system,
            style=state.style,
            layout_tree=state.layout_tree,
            layout_html=state.layout_html,
            html_url=state.html_url,
            render_result=state.render_result,
            critiques=state.feedback_history,
        )

    def _finalize_response(self, state: GraphState) -> GenerateResponse:
        response = self.build_response(state)
        latest = state.feedback_history[-1] if state.feedback_history else None
        return response.model_copy(update={
            "warnings": list(state.warnings),
            "critiques": list(state.feedback_history),
            "score": latest.score if latest else response.score,
        })
