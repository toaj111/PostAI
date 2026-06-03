"""Phase 2: graph runner produces HTML-based posters."""

from app.orchestration.graph_runner import GraphRunner
from app.render.asset_store import AssetStore
from app.schemas.layout import CanvasSpec
from app.schemas.state import GraphState


async def test_graph_runner_produces_final_output(tmp_path):
    """Full pipeline with all fallbacks produces a valid HTML-rendered poster."""
    state = GraphState(
        user_prompt="制作一张科技风 AI 会议海报",
        canvas=CanvasSpec(width=512, height=768),
        max_iterations=2,
    )
    response = await GraphRunner(asset_store=AssetStore(tmp_path, "/assets")).run(state)
    assert response.job_id == state.job_id
    assert response.final_image
    assert response.image_url
    assert response.score is not None
    # Phase 2: layout_html is set by the layout planner.
    assert response.layout_html is not None
    assert "<!DOCTYPE html>" in response.layout_html
    assert response.content_expansion is not None
    assert state.content_expansion is not None
    assert response.content_expansion.poster_type == state.content_expansion.poster_type
