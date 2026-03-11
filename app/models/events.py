from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any
from typing import TypeAlias

from pydantic import BaseModel, Field, ValidationError


class EventType(str, Enum):
    PING = "ping"
    PONG = "pong"
    TEXT = "text"
    AUDIO_CHUNK = "audio.chunk"
    ASSISTANT_TEXT = "assistant.text"
    ASSISTANT_AUDIO = "assistant.audio"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    SESSION_STARTED = "session.started"
    ERROR = "error"


class ClientEvent(BaseModel):
    type: str = Field(default=EventType.TEXT.value)
    payload: dict[str, Any] = Field(default_factory=dict)


class ServerEvent(BaseModel):
    type: str
    session_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AudioChunkMessage(BaseModel):
    type: str = Field(default="audio_chunk")
    format: str = Field(default="pcm16")
    sample_rate: int = Field(default=16000, gt=0)
    data: str


class ControlMessage(BaseModel):
    type: str = Field(default="control")
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)


IncomingMessage: TypeAlias = AudioChunkMessage | ControlMessage


def parse_client_event(raw_text: str) -> ClientEvent:
    text = raw_text.strip()
    if not text:
        return ClientEvent(type=EventType.TEXT.value, payload={"text": ""})

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ClientEvent(type=EventType.TEXT.value, payload={"text": raw_text})

    if not isinstance(payload, dict):
        return ClientEvent(
            type=EventType.ERROR.value,
            payload={"detail": "event must be a JSON object"},
        )

    try:
        return ClientEvent.model_validate(payload)
    except ValidationError as exc:
        return ClientEvent(
            type=EventType.ERROR.value,
            payload={
                "detail": "invalid_event_schema",
                "errors": exc.errors(),
            },
        )


def parse_ws_message(raw_text: str) -> IncomingMessage:
    text = raw_text.strip()
    if not text:
        raise ValueError("message is empty")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("message must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("message must be a JSON object")

    message_type = payload.get("type")
    try:
        if message_type == "audio_chunk":
            return AudioChunkMessage.model_validate(payload)
        if message_type == "control":
            return ControlMessage.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid message schema: {exc.errors()}") from exc

    raise ValueError(f"unsupported message type: {message_type!r}")
