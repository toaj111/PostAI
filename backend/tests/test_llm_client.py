import pytest
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.errors import SchemaParseError
from app.core.llm_client import StructuredLLMClient


class SampleOutput(BaseModel):
    name: str


def test_validate_json_parses_model():
    client = StructuredLLMClient(api_key=None, base_url=None, model="mock")
    parsed = client.validate_json('{"name":"poster"}', SampleOutput)
    assert parsed.name == "poster"


def test_validate_json_rejects_invalid_json():
    client = StructuredLLMClient(api_key=None, base_url=None, model="mock")
    with pytest.raises(SchemaParseError):
        client.validate_json("not-json", SampleOutput)


def test_json_object_mode_adds_schema_hint():
    client = StructuredLLMClient(api_key="k", base_url="https://example.test", model="m", response_format="json_object")
    messages = [{"role": "user", "content": "hello"}]
    hinted = client._messages_with_schema_hint(messages, SampleOutput)
    assert hinted[0]["role"] == "system"
    assert "JSON Schema" in hinted[0]["content"]
    assert hinted[1:] == messages


def test_apply_token_limit_uses_structured_and_raw_limits():
    client = StructuredLLMClient(
        api_key="k",
        base_url="https://example.test",
        model="m",
        max_tokens=1234,
        raw_max_tokens=6789,
    )
    structured_payload = {}
    raw_payload = {}
    client._apply_token_limit(structured_payload, force_raw=False)
    client._apply_token_limit(raw_payload, force_raw=True)
    assert structured_payload["max_tokens"] == 1234
    assert raw_payload["max_tokens"] == 6789


def test_client_accepts_separate_structured_and_raw_timeouts():
    client = StructuredLLMClient(
        api_key="k",
        base_url="https://example.test",
        model="m",
        timeout=12,
        raw_timeout=345,
    )
    assert client.timeout == 12
    assert client.raw_timeout == 345


def test_apply_token_limit_skips_non_positive_values():
    client = StructuredLLMClient(
        api_key="k",
        base_url="https://example.test",
        model="m",
        max_tokens=0,
        raw_max_tokens=-1,
    )
    structured_payload = {}
    raw_payload = {}
    client._apply_token_limit(structured_payload, force_raw=False)
    client._apply_token_limit(raw_payload, force_raw=True)
    assert "max_tokens" not in structured_payload
    assert "max_tokens" not in raw_payload


def test_settings_loads_dotenv_values(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_MODEL=dotenv-model\nVISION_MODEL=dotenv-vision\nLLM_RAW_MAX_TOKENS=9999\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("LLM_RAW_MAX_TOKENS", raising=False)

    from app.core.config import load_environment

    load_environment(env_file, override=True)
    settings = get_settings()
    assert settings.llm_model == "dotenv-model"
    assert settings.vision_model == "dotenv-vision"
    assert settings.llm_raw_max_tokens == 9999
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("LLM_RAW_MAX_TOKENS", raising=False)


def test_settings_reads_model_fallback_flag(monkeypatch):
    monkeypatch.setenv("ALLOW_MODEL_FALLBACK", "false")
    assert get_settings().allow_model_fallback is False
    monkeypatch.setenv("ALLOW_MODEL_FALLBACK", "true")
    assert get_settings().allow_model_fallback is True
