from httpx import ASGITransport, AsyncClient

from app.main import app
from app.render.asset_store import AssetStore


_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2X8mQAAAAASUVORK5CYII="
)


async def test_generate_endpoint_returns_image():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/generate",
            json={"prompt": "制作一张科技风 AI 会议海报", "width": 512, "height": 768, "max_iterations": 1},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["final_image"]
    assert payload["image_url"].startswith("/assets/")


async def test_generate_endpoint_accepts_reference_images():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/generate",
            json={
                "prompt": "制作一张科技风 AI 会议海报",
                "width": 512,
                "height": 768,
                "max_iterations": 1,
                "reference_images": [
                    {
                        "url": "https://images.unsplash.com/photo-1518770660439-4636190af475",
                        "description": "蓝色科技感人物特写",
                    },
                    {
                        "url": "https://images.unsplash.com/photo-1461749280684-dccba630e2f6",
                        "description": "深色背景与屏幕光效",
                    },
                ],
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"]


async def test_generate_endpoint_rejects_invalid_reference_image_url():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/generate",
            json={
                "prompt": "制作一张科技风 AI 会议海报",
                "reference_images": [
                    {"url": "ftp://example.com/image.png", "description": "invalid"}
                ],
            },
        )
    assert response.status_code == 422


    async def test_upload_reference_image_returns_public_url(tmp_path, monkeypatch):
        from app.api import routes_generate

        monkeypatch.setattr(routes_generate, "_asset_store", lambda: AssetStore(tmp_path, "/assets"))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/reference-images/upload",
                json={
                    "filename": "local.png",
                    "mime_type": "image/png",
                    "data_url": _TINY_PNG_DATA_URL,
                    "description": "local upload",
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["url"].startswith("http://test/assets/reference_uploads/")
        assert payload["filename"] == "local.png"


async def test_generate_stream_endpoint_emits_sse():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/generate/stream",
            json={"prompt": "招聘海报", "width": 512, "height": 768, "max_iterations": 1},
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: final_output" in response.text


async def test_generate_endpoint_returns_502_on_graph_error(monkeypatch):
    from app.orchestration.graph_runner import GraphRunner

    async def fail(self, state):
        raise RuntimeError("model failed")

    monkeypatch.setattr(GraphRunner, "run", fail)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/generate", json={"prompt": "x"})
    assert response.status_code == 502
    assert response.json()["detail"] == "model failed"
