from __future__ import annotations

import json
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.errors import LLMCallError, SchemaParseError
from app.core.logging import get_logger
from app.orchestration.retry import retry_async

logger = get_logger("llm_client")

ModelT = TypeVar("ModelT", bound=BaseModel)

MAX_MSG_LOG_CHARS = 1500


def _summarize_messages(messages: list[dict[str, Any]]) -> str:
    """Brief summary: role + first chars of content for each message."""
    items = []
    for m in messages:
        role = m.get("role", "?")
        content = str(m.get("content", ""))
        if len(content) > MAX_MSG_LOG_CHARS:
            content = content[:MAX_MSG_LOG_CHARS] + f"... [truncated, total {len(content)} chars]"
        items.append(f"[{role}] {content}")
    return "\n".join(items)


def _summarize_vision_messages(messages: list[dict[str, Any]]) -> str:
    """Like _summarize_messages but marks image content compactly."""
    items = []
    for m in messages:
        role = m.get("role", "?")
        raw = m.get("content", "")
        # Vision messages may have content as a list of parts
        if isinstance(raw, list):
            parts: list[str] = []
            for part in raw:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        url = str(part.get("image_url", {}).get("url", ""))
                        parts.append("[image:" + (url[:40] + "..." if len(url) > 40 else url) + "]")
                    elif part.get("type") == "text":
                        text = str(part.get("text", ""))
                        if len(text) > MAX_MSG_LOG_CHARS:
                            text = text[:MAX_MSG_LOG_CHARS] + "..."
                        parts.append(text)
            items.append(f"[{role}] {' | '.join(parts)}")
        else:
            content = str(raw)
            if len(content) > MAX_MSG_LOG_CHARS:
                content = content[:MAX_MSG_LOG_CHARS] + f"... [truncated, total {len(content)} chars]"
            items.append(f"[{role}] {content}")
    return "\n".join(items)


class StructuredLLMClient:
    """Thin placeholder for future OpenAI-compatible structured calls.

    This client targets OpenAI-compatible ``/chat/completions`` APIs. The
    built-in agents still work without API keys; this class is the replacement
    boundary for real Content/Style/Layout/VLM agents.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        model: str,
        timeout: float = 60,
        response_format: str = "json_schema",
        temperature: float | None = None,
        raw_temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.response_format = response_format
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.raw_temperature = settings.llm_raw_temperature if raw_temperature is None else raw_temperature

    async def parse(self, *, messages: list[dict[str, Any]], response_model: type[ModelT]) -> ModelT:
        if not self.api_key or not self.base_url or self.model.startswith("mock-"):
            raise LLMCallError("LLM provider is not configured")

        async def _call() -> ModelT:
            content = await self._chat_completion(
                messages=self._messages_with_schema_hint(messages, response_model),
                response_model=response_model,
            )
            return self.validate_json(content, response_model)

        return await retry_async(_call, attempts=3, delay_seconds=0.4, exceptions=(LLMCallError, SchemaParseError))

    def _messages_with_schema_hint(
        self,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
    ) -> list[dict[str, Any]]:
        if self.response_format != "json_object":
            return messages
        schema_hint = (
            "You must return a single JSON object that validates against this JSON Schema. "
            "Do not return markdown, explanations, layout canvas specs, or any fields outside the schema. "
            f"Schema: {json.dumps(response_model.model_json_schema(), ensure_ascii=False)}"
        )
        return [{"role": "system", "content": schema_hint}, *messages]

    async def _chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        force_raw: bool = False,
    ) -> str:
        """Call the chat API.  Set *force_raw* to ``True`` to skip
        ``response_format`` constraints — needed when the expected output
        is free-form text (e.g. HTML) rather than JSON.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.raw_temperature if force_raw else self.temperature,
        }
        if not force_raw and self.response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            }
        elif not force_raw and self.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        fmt_display = "raw" if force_raw else self.response_format
        logger.info(
            "--> LLM CALL  | %s | model=%s | format=%s | temp=%.2f | messages=\n%s",
            url,
            self.model,
            fmt_display,
            payload["temperature"],
            _summarize_messages(messages),
        )

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("<-- LLM FAIL | %s | duration=%.1fms | error=%s", url, elapsed, exc)
            raise LLMCallError(f"LLM request failed: {exc}") from exc

        elapsed = (time.perf_counter() - start) * 1000
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(
                "<-- LLM ERROR | %s | status=%d | duration=%.1fms | body=%s",
                url,
                response.status_code,
                elapsed,
                json.dumps(data, ensure_ascii=False),
            )
            raise LLMCallError("LLM response did not contain choices[0].message.content") from exc

        if not isinstance(content, str) or not content.strip():
            logger.error("<-- LLM EMPTY | %s | status=%d | duration=%.1fms", url, response.status_code, elapsed)
            raise LLMCallError("LLM response content is empty")

        content_display = content if len(content) <= 2000 else content[:2000] + f"... [truncated, total {len(content)} chars]"
        logger.info("<-- LLM OK   | %s | status=%d | duration=%.1fms | content=%s", url, response.status_code, elapsed, content_display)
        return content

    # ── Vision model methods (with thinking/reasoning support) ──

    async def parse_vision(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        enable_thinking: bool = True,
        thinking_budget: int = 8192,
    ) -> tuple[ModelT, str]:
        """Parse structured output from a vision-capable model.

        Returns ``(parsed_model, reasoning_content)`` where *reasoning_content*
        is the model's chain-of-thought thinking about the image.  Only meaningful
        when *enable_thinking* is ``True`` and the provider supports it.

        Unlike ``parse()`` this method does **not** inject a schema-hint system
        message nor send ``response_format`` — vision models with thinking mode
        are guided purely by the caller's prompt so the thinking stream is
        unconstrained.
        """
        if not self.api_key or not self.base_url or self.model.startswith("mock-"):
            raise LLMCallError("Vision provider is not configured")

        async def _call() -> tuple[ModelT, str]:
            content, reasoning = await self._vision_completion(
                messages=messages,
                response_model=response_model,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
            return self.validate_json(content, response_model), reasoning

        return await retry_async(
            _call, attempts=3, delay_seconds=0.4, exceptions=(LLMCallError, SchemaParseError)
        )

    async def _vision_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        enable_thinking: bool = True,
        thinking_budget: int = 8192,
    ) -> tuple[str, str]:
        """Call the vision model and return ``(content, reasoning_content)``.

        When *enable_thinking* is ``True`` the payload includes
        ``extra_body.enable_thinking`` so providers that support chain-of-thought
        (e.g. Qwen VL) will emit reasoning tokens.  No ``response_format`` is
        sent because many vision models reject it together with thinking mode.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        # When thinking is disabled we can use strict response_format as a
        # fallback; otherwise we trust the prompt to produce valid JSON.
        if enable_thinking:
            payload["extra_body"] = {
                "enable_thinking": True,
                "thinking_budget": thinking_budget,
            }
        elif self.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        logger.info(
            "--> VISION CALL | %s | model=%s | thinking=%s | messages=\n%s",
            url,
            self.model,
            "on" if enable_thinking else "off",
            _summarize_vision_messages(messages),
        )

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("<-- VISION FAIL | %s | duration=%.1fms | error=%s", url, elapsed, exc)
            raise LLMCallError(f"Vision request failed: {exc}") from exc

        elapsed = (time.perf_counter() - start) * 1000
        data = response.json()

        # Extract the assistant message — for thinking models this carries both
        # the final content and the reasoning chain.
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(
                "<-- VISION ERROR | %s | status=%d | duration=%.1fms | body=%s",
                url,
                response.status_code,
                elapsed,
                json.dumps(data, ensure_ascii=False),
            )
            raise LLMCallError("Vision response did not contain choices[0].message") from exc

        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")

        if not isinstance(content, str) or not content.strip():
            logger.error("<-- VISION EMPTY | %s | status=%d | duration=%.1fms", url, response.status_code, elapsed)
            raise LLMCallError("Vision response content is empty")

        reasoning_display = ""
        if reasoning:
            reasoning_display = reasoning if len(reasoning) <= 1000 else reasoning[:1000] + f"... [truncated, total {len(reasoning)} chars]"
            logger.info("<-- VISION REASONING | %s | %s", url, reasoning_display)

        content_display = content if len(content) <= 2000 else content[:2000] + f"... [truncated, total {len(content)} chars]"
        logger.info("<-- VISION OK   | %s | status=%d | duration=%.1fms | content=%s", url, response.status_code, elapsed, content_display)
        return content, reasoning

    def validate_json(self, content: str, response_model: type[ModelT]) -> ModelT:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise SchemaParseError(f"LLM output is not valid JSON: {exc}") from exc
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            raise SchemaParseError(f"LLM output failed schema validation: {exc}") from exc

    def validate_payload(self, payload: dict[str, Any], response_model: type[ModelT]) -> ModelT:
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            raise SchemaParseError(f"LLM output failed schema validation: {exc}") from exc
