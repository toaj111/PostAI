"""Phase 2 router tests — natural-language suggestions instead of adjustment vectors."""

from app.orchestration.router import RouteAction, route_after_critique
from app.schemas.agents import CritiqueResult
from app.schemas.state import GraphState


def test_router_requires_min_iteration_before_final():
    """With min_iterations=1, first iteration won't finalise on score alone."""
    state = GraphState(user_prompt="poster", min_iterations=1)
    state.feedback_history.append(CritiqueResult(score=90, passed=True, reasoning="ok"))
    # iteration_count=0 < min_iterations=1 → continue.
    assert route_after_critique(state).action == RouteAction.layout


def test_router_revision_focus_final_still_honours_min_iteration():
    state = GraphState(user_prompt="poster", min_iterations=1, max_iterations=2)
    state.feedback_history.append(
        CritiqueResult(score=90, passed=True, reasoning="ok", revision_focus="final")
    )

    assert route_after_critique(state).action == RouteAction.layout


def test_router_finishes_when_score_passes_after_min_iteration():
    state = GraphState(user_prompt="poster", min_iterations=1)
    state.iteration_count = 1
    state.feedback_history.append(CritiqueResult(score=90, passed=True, reasoning="ok"))
    assert route_after_critique(state).action == RouteAction.final


def test_router_finishes_immediately_when_min_iterations_zero():
    """With min_iterations=0 (default), high score finalises immediately."""
    state = GraphState(user_prompt="poster", min_iterations=0)
    state.feedback_history.append(CritiqueResult(score=90, passed=True, reasoning="ok"))
    assert route_after_critique(state).action == RouteAction.final


def test_router_sends_background_issue_to_style():
    state = GraphState(user_prompt="poster")
    state.feedback_history.append(
        CritiqueResult(
            score=70, passed=False, reasoning="busy background",
            issues=["Background pattern is too distracting"],
            suggestions=["Use a simpler gradient background"],
            revision_focus="style",
        )
    )
    assert route_after_critique(state).action == RouteAction.style


def test_router_sends_layout_issues_to_layout():
    state = GraphState(user_prompt="poster")
    state.feedback_history.append(
        CritiqueResult(
            score=65, passed=False, reasoning="spacing problems",
            issues=["Title is too close to the top edge"],
            suggestions=["Move the title down by about 0.05"],
            revision_focus="layout",
        )
    )
    assert route_after_critique(state).action == RouteAction.layout


def test_router_sends_content_focus_to_content():
    """Phase 5: revision_focus='content' routes to ContentExtractor."""
    state = GraphState(user_prompt="poster")
    state.feedback_history.append(
        CritiqueResult(
            score=55, passed=False, reasoning="wrong headline text",
            issues=["Headline text does not match the user prompt"],
            revision_focus="content",
        )
    )
    assert route_after_critique(state).action == RouteAction.content


def test_router_sends_render_focus_to_layout():
    """Phase 5: revision_focus='render' routes to layout (re-render after fix)."""
    state = GraphState(user_prompt="poster")
    state.feedback_history.append(
        CritiqueResult(
            score=40, passed=False, reasoning="broken CSS",
            issues=["Font failed to load, text invisible"],
            revision_focus="render",
        )
    )
    assert route_after_critique(state).action == RouteAction.layout


def test_router_sends_final_focus_even_with_low_score():
    """Phase 5: revision_focus='final' overrides score checks."""
    state = GraphState(user_prompt="poster")
    state.feedback_history.append(
        CritiqueResult(
            score=60, passed=True, reasoning="good enough for this poster type",
            revision_focus="final",
        )
    )
    assert route_after_critique(state).action == RouteAction.final


def test_router_finishes_when_max_iterations_reached():
    state = GraphState(user_prompt="poster", max_iterations=2)
    state.iteration_count = 2
    state.feedback_history.append(CritiqueResult(score=50, passed=False, reasoning="not there yet"))
    assert route_after_critique(state).action == RouteAction.final


def test_router_finishes_when_score_stagnates():
    state = GraphState(user_prompt="poster")
    state.iteration_count = 1
    state.feedback_history.append(CritiqueResult(score=60, passed=False, reasoning="first"))
    state.feedback_history.append(CritiqueResult(score=55, passed=False, reasoning="worse"))
    assert route_after_critique(state).action == RouteAction.final
