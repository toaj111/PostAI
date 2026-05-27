"""Routing logic — Phase 5 (revision_focus-driven).

Decides whether the iteration loop should finalise the poster or run another
round of layout / style / content planning.  The router now reads
``revision_focus`` from the latest ``CritiqueResult`` instead of scanning
natural-language keywords.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from app.schemas.state import GraphState


class RouteAction(str, Enum):
    final = "final"
    layout = "layout"
    style = "style"
    content = "content"
    render = "render"


class RouteDecision(BaseModel):
    action: RouteAction
    reason: str


def route_after_critique(state: GraphState) -> RouteDecision:
    """Decide the next pipeline action based on the most recent critique.

    Phase 5: reads ``revision_focus`` from the latest CritiqueResult.
    Falls back to layout when no critique is available.
    """
    if not state.feedback_history:
        return RouteDecision(action=RouteAction.layout, reason="No critique is available yet.")

    latest = state.feedback_history[-1]
    min_ok = state.iteration_count >= state.min_iterations
    has_iteration_budget = state.iteration_count + 1 < state.max_iterations

    # Honour min_iterations before every possible finalisation path, including
    # VLM's revision_focus="final". With min_iterations=1 the first critique
    # should force at least one more layout/render pass when budget remains.
    if not min_ok:
        if not has_iteration_budget:
            return RouteDecision(action=RouteAction.final, reason="Max iterations reached before min_iterations could be satisfied.")
        focus = latest.revision_focus
        if focus == "style":
            return RouteDecision(action=RouteAction.style, reason="Minimum iterations not met; applying requested style revision.")
        if focus == "content":
            return RouteDecision(action=RouteAction.content, reason="Minimum iterations not met; applying requested content revision.")
        return RouteDecision(action=RouteAction.layout, reason="Minimum iterations not met; run another layout/render review.")

    # ── Score-based early exit ──
    if latest.score >= state.target_score:
        return RouteDecision(action=RouteAction.final, reason="Target score reached.")
    if latest.passed and latest.score >= state.target_score - 10:
        return RouteDecision(action=RouteAction.final, reason="VLM passed and score is close to target.")

    # ── Iteration budget exhausted ──
    if state.iteration_count + 1 >= state.max_iterations:
        return RouteDecision(action=RouteAction.final, reason="Max iterations reached.")

    # ── Score stagnation ──
    if len(state.feedback_history) >= 2 and latest.score <= state.feedback_history[-2].score:
        return RouteDecision(action=RouteAction.final, reason="Score did not improve in the latest iteration.")

    # ── Phase 5: revision_focus-driven routing ──
    focus = latest.revision_focus

    if focus == "final":
        return RouteDecision(action=RouteAction.final, reason="VLM marked as final.")
    if focus == "style":
        return RouteDecision(action=RouteAction.style, reason="VLM requested style revision.")
    if focus == "content":
        return RouteDecision(action=RouteAction.content, reason="VLM requested content revision.")
    if focus == "render":
        return RouteDecision(action=RouteAction.layout, reason="VLM requested render fix — re-layout to address.")

    # Default: layout revision.
    return RouteDecision(action=RouteAction.layout, reason="Apply layout improvements based on feedback.")
