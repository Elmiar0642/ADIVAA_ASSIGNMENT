from dataclasses import dataclass
import json
import os
import re
from typing import Any

import aiohttp

from app.core.logging import get_logger


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class LLMResult:
    text: str | None = None
    tool_call: ToolCall | None = None
    raw_response: dict[str, Any] | None = None


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 20.0,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = os.getenv("OPENAI_LLM_MODEL", model)
        self._base_url = os.getenv("OPENAI_BASE_URL", base_url).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._logger = get_logger(__name__)
        self._fallback_model = os.getenv("OPENAI_LLM_FALLBACK_MODEL", "gpt-4o-mini")

    async def reason(self, transcript_text: str) -> LLMResult:
        prompt = transcript_text.strip()
        if not prompt:
            return LLMResult(text="Please say that again.")

        heuristic_tool_call = self._heuristic_tool_call(prompt)
        if heuristic_tool_call is not None:
            return LLMResult(tool_call=heuristic_tool_call)

        if not self._api_key:
            # Graceful local fallback when key is missing.
            return LLMResult(text=f"Echo response: {prompt}")

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a helpful voice assistant."},
                {"role": "user", "content": prompt},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "play_audio",
                        "description": "Play a known track for the user.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "track": {
                                    "type": "string",
                                    "description": "Track name to play.",
                                }
                            },
                            "required": ["track"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self._base_url}/chat/completions"

        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            data = await self._call_chat_completion(session, endpoint, headers, payload)
            if data is None and self._fallback_model and self._fallback_model != self._model:
                retry_payload = dict(payload)
                retry_payload["model"] = self._fallback_model
                data = await self._call_chat_completion(session, endpoint, headers, retry_payload)
                if data is not None:
                    self._logger.warning(
                        "llm_fallback_model_used",
                        extra={
                            "primary_model": self._model,
                            "fallback_model": self._fallback_model,
                        },
                    )

        if data is None:
            return LLMResult(text=f"Echo response: {prompt}")

        return self._parse_chat_completion(data)

    async def generate_response(self, prompt: str) -> str:
        result = await self.reason(prompt)
        if result.text:
            return result.text
        if result.tool_call:
            return f"Calling tool: {result.tool_call.name}"
        return "I could not generate a response."

    def _parse_chat_completion(self, payload: dict[str, Any]) -> LLMResult:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return LLMResult(text="I could not generate a response.", raw_response=payload)

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return LLMResult(text="I could not generate a response.", raw_response=payload)

        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            return LLMResult(text="I could not generate a response.", raw_response=payload)

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            first_tool_call = tool_calls[0]
            if isinstance(first_tool_call, dict):
                function_payload = first_tool_call.get("function", {})
                if isinstance(function_payload, dict):
                    name = function_payload.get("name")
                    raw_arguments = function_payload.get("arguments", "{}")
                    if isinstance(name, str) and name:
                        parsed_arguments = self._safe_parse_json(raw_arguments)
                        if not isinstance(parsed_arguments, dict):
                            parsed_arguments = {"track": str(raw_arguments)}
                        return LLMResult(
                            tool_call=ToolCall(name=name, arguments=parsed_arguments),
                            raw_response=payload,
                        )

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return LLMResult(text=content.strip(), raw_response=payload)

        return LLMResult(text="I could not generate a response.", raw_response=payload)

    def _safe_parse_json(self, raw: Any) -> Any:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"track": raw}

    async def _call_chat_completion(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                if response.status >= 400:
                    body = await response.text()
                    self._logger.error(
                        "llm_request_failed",
                        extra={
                            "status": response.status,
                            "body": body[:1000],
                            "model": payload.get("model"),
                        },
                    )
                    return None
                return await response.json()
        except Exception as exc:
            self._logger.exception(
                "llm_request_exception",
                extra={
                    "error": str(exc),
                    "model": payload.get("model"),
                },
            )
            return None

    def _heuristic_tool_call(self, prompt: str) -> ToolCall | None:
        normalized = " ".join(prompt.strip().split())
        if not normalized:
            return None
        match = re.search(r"\bplay\s+track\s+([a-zA-Z0-9_.-]+)\b", normalized, flags=re.IGNORECASE)
        if not match:
            return None
        track = match.group(1).strip()
        if not track:
            return None
        return ToolCall(
            name="play_audio",
            arguments={
                "track": track,
                "stream_audio": True,
                "format": "pcm16",
                "sample_rate": 16000,
            },
        )
