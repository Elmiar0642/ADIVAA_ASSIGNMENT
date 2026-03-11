import asyncio
import base64
import inspect
import json
import os
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import websockets
from websockets.exceptions import ConnectionClosed

from app.core.logging import get_logger

TranscriptCallback = Callable[[str, bool, dict[str, Any]], Awaitable[None] | None]
ResponseCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
ErrorCallback = Callable[[str, dict[str, Any] | None], Awaitable[None] | None]

_PARTIAL_TRANSCRIPT_TYPES = {
    "conversation.item.input_audio_transcription.delta",
    "response.audio_transcript.delta",
    "transcript.partial",
}
_FINAL_TRANSCRIPT_TYPES = {
    "conversation.item.input_audio_transcription.completed",
    "response.audio_transcript.done",
    "transcript.final",
}
_RESPONSE_TYPES = {
    "response.output_text.delta",
    "response.output_text.done",
    "response.text.delta",
    "response.text.done",
    "response.done",
    "response.completed",
    "model.response",
}
_SESSION_CONFIG_EVENT_TYPES = {
    "session.created",
    "session.updated",
    "transcription_session.created",
    "transcription_session.updated",
}


class RealtimeClient:
    def __init__(
        self,
        api_key: str | None = None,
        ws_url: str | None = None,
        model: str | None = None,
        reconnect_attempts: int = 5,
        reconnect_base_delay_seconds: float = 1.0,
        reconnect_max_delay_seconds: float = 12.0,
        on_transcript: TranscriptCallback | None = None,
        on_response: ResponseCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        self._logger = get_logger(__name__)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._ws_url = ws_url or os.getenv("OPENAI_REALTIME_URL", "wss://api.openai.com/v1/realtime")
        self._model = model or os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
        self._beta_header = os.getenv("OPENAI_REALTIME_BETA", "realtime=v1")
        self._transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
        self._open_timeout = float(os.getenv("OPENAI_REALTIME_OPEN_TIMEOUT_SECONDS", "15"))
        self._close_timeout = float(os.getenv("OPENAI_REALTIME_CLOSE_TIMEOUT_SECONDS", "10"))
        self._max_message_size = int(os.getenv("OPENAI_REALTIME_MAX_MESSAGE_SIZE_BYTES", str(8 * 1024 * 1024)))

        self._reconnect_attempts = reconnect_attempts
        self._reconnect_base_delay_seconds = reconnect_base_delay_seconds
        self._reconnect_max_delay_seconds = reconnect_max_delay_seconds

        self._on_transcript = on_transcript
        self._on_response = on_response
        self._on_error = on_error

        self._ws: Any | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._closing = False
        self._mock_mode = False
        self._mock_frames_since_final = 0
        self._mock_bytes_since_final = 0

    @property
    def connected(self) -> bool:
        if self._mock_mode:
            return True
        if self._ws is None:
            return False
        return not bool(getattr(self._ws, "closed", False))

    def set_callbacks(
        self,
        on_transcript: TranscriptCallback | None = None,
        on_response: ResponseCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        if on_transcript is not None:
            self._on_transcript = on_transcript
        if on_response is not None:
            self._on_response = on_response
        if on_error is not None:
            self._on_error = on_error

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._mock_mode:
                return
            if self._ws is not None:
                return
            self._closing = False
            if not self._api_key:
                self._mock_mode = True
                self._logger.warning(
                    "realtime_client_mock_mode_enabled",
                    extra={
                        "reason": "missing_openai_api_key",
                        "event_type": "realtime.mock_mode.enabled",
                    },
                )
                return
            await self._connect_with_retries()
            await self._initialize_session()
            if self._recv_task is None or self._recv_task.done():
                self._recv_task = asyncio.create_task(self._receive_loop(), name="openai-realtime-recv")

    async def disconnect(self) -> None:
        self._closing = True
        self._mock_mode = False
        self._mock_frames_since_final = 0
        self._mock_bytes_since_final = 0

        recv_task = self._recv_task
        self._recv_task = None

        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close(code=1000, reason="client_shutdown")
            except Exception:
                self._logger.exception("realtime_ws_close_failed")

        if recv_task is not None and not recv_task.done():
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)

        self._logger.info("realtime_client_disconnected")

    async def send_audio_frame(
        self,
        audio: bytes,
        *,
        audio_format: str = "pcm16",
        sample_rate: int = 16000,
    ) -> dict[str, Any]:
        audio_base64 = base64.b64encode(audio).decode("ascii")
        event = {
            "type": "input_audio_buffer.append",
            "audio": audio_base64,
        }
        return await self.send_event(event)

    async def send_event(self, event: dict[str, Any]) -> dict[str, Any]:
        if self._closing:
            return {"status": "closed"}

        if self._mock_mode:
            return await self._send_event_mock(event)

        if self._ws is None:
            await self.connect()

        payload = json.dumps(event, separators=(",", ":"))
        async with self._send_lock:
            ws = self._ws
            if ws is None:
                await self._reconnect("send_without_socket")
                ws = self._ws
            if ws is None:
                raise RuntimeError("realtime websocket is not connected")

            try:
                await ws.send(payload)
            except ConnectionClosed as exc:
                self._logger.warning(
                    "realtime_ws_send_closed",
                    extra={
                        "code": getattr(exc, "code", None),
                        "reason": getattr(exc, "reason", None),
                    },
                )
                self._ws = None
                await self._reconnect("send_connection_closed")
                ws = self._ws
                if ws is None:
                    raise RuntimeError("realtime websocket reconnection failed") from exc
                await ws.send(payload)

        self._logger.info("realtime_event_sent", extra={"event_type": event.get("type")})
        return {"status": "sent", "event_type": event.get("type")}

    async def _initialize_session(self) -> None:
        ws = self._ws
        if ws is None:
            return
        init_event = {
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": self._transcription_model},
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": 500,
                    "prefix_padding_ms": 300,
                },
            },
        }
        payload = json.dumps(init_event, separators=(",", ":"))
        try:
            await ws.send(payload)
            self._logger.info(
                "realtime_session_initialized",
                extra={
                    "event_type": "realtime.session.initialized",
                    "transcription_model": self._transcription_model,
                },
            )
        except Exception as exc:
            await self._emit_error("realtime_session_init_failed", {"error": str(exc)})

    async def _send_event_mock(self, event: dict[str, Any]) -> dict[str, Any]:
        event_type = str(event.get("type", ""))
        if event_type != "input_audio_buffer.append":
            self._logger.info(
                "realtime_event_sent_mock",
                extra={"event_type": event_type, "mock_mode": True},
            )
            return {"status": "sent", "event_type": event_type, "mock_mode": True}

        audio_b64 = event.get("audio", "")
        chunk_bytes = 0
        if isinstance(audio_b64, str) and audio_b64:
            try:
                chunk_bytes = len(base64.b64decode(audio_b64, validate=False))
            except Exception:
                chunk_bytes = 0
        self._mock_frames_since_final += 1
        self._mock_bytes_since_final += max(chunk_bytes, 0)

        # Emit periodic partials and finals to keep the local pipeline active.
        if self._mock_frames_since_final % 4 == 0:
            await self._invoke_callback(
                self._on_transcript,
                "listening...",
                False,
                {"type": "transcript.partial", "mock": True},
            )
        if self._mock_frames_since_final >= 10:
            final_text = f"transcribed {self._mock_bytes_since_final} bytes of audio"
            await self._invoke_callback(
                self._on_transcript,
                final_text,
                True,
                {"type": "transcript.final", "mock": True},
            )
            self._mock_frames_since_final = 0
            self._mock_bytes_since_final = 0

        self._logger.info(
            "realtime_event_sent_mock",
            extra={"event_type": event_type, "chunk_bytes": chunk_bytes, "mock_mode": True},
        )
        return {"status": "sent", "event_type": event_type, "mock_mode": True}

    async def _connect_with_retries(self) -> None:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required for Realtime API")

        ws_url = self._resolved_ws_url()
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._beta_header:
            headers["OpenAI-Beta"] = self._beta_header

        last_error: Exception | None = None
        for attempt in range(1, self._reconnect_attempts + 2):
            try:
                self._ws = await self._ws_connect(ws_url, headers)
                self._logger.info(
                    "realtime_client_connected",
                    extra={"url": ws_url, "attempt": attempt},
                )
                return
            except Exception as exc:
                last_error = exc
                await self._emit_error(
                    "realtime_connect_failed",
                    {"attempt": attempt, "error": str(exc)},
                )

                if attempt > self._reconnect_attempts:
                    break

                delay_seconds = min(
                    self._reconnect_base_delay_seconds * (2 ** (attempt - 1)),
                    self._reconnect_max_delay_seconds,
                )
                self._logger.warning(
                    "realtime_connect_retry_scheduled",
                    extra={"attempt": attempt, "sleep_seconds": delay_seconds},
                )
                await asyncio.sleep(delay_seconds)

        raise RuntimeError("unable to connect to OpenAI Realtime API") from last_error

    async def _reconnect(self, reason: str) -> None:
        if self._closing:
            return
        async with self._connect_lock:
            if self._closing:
                return
            if self._ws is not None:
                return
            self._logger.warning("realtime_reconnect_start", extra={"reason": reason})
            await self._connect_with_retries()

    async def _receive_loop(self) -> None:
        while not self._closing:
            ws = self._ws
            if ws is None:
                try:
                    await self._reconnect("receive_loop_no_socket")
                except Exception as exc:
                    await self._emit_error("realtime_reconnect_failed", {"error": str(exc)})
                    return
                continue

            try:
                raw = await ws.recv()
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as exc:
                self._logger.warning(
                    "realtime_ws_closed",
                    extra={
                        "code": getattr(exc, "code", None),
                        "reason": getattr(exc, "reason", None),
                    },
                )
                self._ws = None
                if self._closing:
                    return
                await self._emit_error(
                    "realtime_connection_closed",
                    {"code": getattr(exc, "code", None), "reason": getattr(exc, "reason", None)},
                )
                continue
            except Exception as exc:
                self._ws = None
                if self._closing:
                    return
                await self._emit_error("realtime_receive_error", {"error": str(exc)})
                continue

            await self._handle_raw_message(raw)

    async def _handle_raw_message(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                await self._emit_error("invalid_utf8_message")
                return

        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            await self._emit_error("invalid_json_message", {"raw": raw[:500]})
            return

        if not isinstance(event, dict):
            await self._emit_error("invalid_event_shape", {"type": type(event).__name__})
            return

        await self._dispatch_event(event)

    async def _dispatch_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))

        if event_type in _SESSION_CONFIG_EVENT_TYPES:
            session_payload = event.get("session")
            self._logger.info(
                "realtime_session_event",
                extra={
                    "event_type": event_type,
                    "session_payload": session_payload if isinstance(session_payload, dict) else {},
                },
            )
            return

        if event_type == "error":
            await self._emit_error("realtime_api_error", event)
            return

        if self._is_transcript_partial(event_type):
            text = self._extract_text(event)
            if text:
                await self._invoke_callback(self._on_transcript, text, False, event)
            return

        if self._is_transcript_final(event_type):
            text = self._extract_text(event)
            if text:
                await self._invoke_callback(self._on_transcript, text, True, event)
            return

        if self._is_response_event(event_type):
            response_payload = {
                "type": event_type,
                "text": self._extract_text(event),
                "event": event,
            }
            await self._invoke_callback(self._on_response, response_payload)

    async def _emit_error(self, message: str, details: dict[str, Any] | None = None) -> None:
        self._logger.error(
            "realtime_client_error",
            extra={"error_message": message, "details": details or {}},
        )
        await self._invoke_callback(self._on_error, message, details)

    async def _invoke_callback(self, callback: Any, *args: Any) -> None:
        if callback is None:
            return
        try:
            result = callback(*args)
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._logger.exception("realtime_callback_failed")

    def _resolved_ws_url(self) -> str:
        parsed = urlparse(self._ws_url)
        query_map = parse_qs(parsed.query)

        if ("model" not in query_map or not query_map["model"]) and self._model:
            query_map["model"] = [self._model]

        encoded_query = urlencode(query_map, doseq=True)
        return urlunparse(parsed._replace(query=encoded_query))

    async def _ws_connect(self, ws_url: str, headers: dict[str, str]) -> Any:
        kwargs = {
            "open_timeout": self._open_timeout,
            "close_timeout": self._close_timeout,
            "ping_interval": 20,
            "ping_timeout": 20,
            "max_size": self._max_message_size,
        }

        try:
            return await websockets.connect(
                ws_url,
                additional_headers=headers,
                **kwargs,
            )
        except TypeError:
            return await websockets.connect(
                ws_url,
                extra_headers=headers,
                **kwargs,
            )

    def _is_transcript_partial(self, event_type: str) -> bool:
        if event_type in _PARTIAL_TRANSCRIPT_TYPES:
            return True
        return "transcript" in event_type and any(token in event_type for token in ("delta", "partial"))

    def _is_transcript_final(self, event_type: str) -> bool:
        if event_type in _FINAL_TRANSCRIPT_TYPES:
            return True
        return "transcript" in event_type and any(token in event_type for token in ("done", "completed", "final"))

    def _is_response_event(self, event_type: str) -> bool:
        if event_type in _RESPONSE_TYPES:
            return True
        return event_type.startswith("response.")

    def _extract_text(self, payload: Any) -> str | None:
        if isinstance(payload, str):
            text = payload.strip()
            return text or None

        if isinstance(payload, dict):
            for key in ("delta", "text", "transcript", "output_text", "content"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            for key in (
                "response",
                "item",
                "message",
                "part",
                "data",
                "result",
                "content_part",
                "audio_transcript",
            ):
                if key in payload:
                    extracted = self._extract_text(payload[key])
                    if extracted:
                        return extracted

            # Avoid recursively searching every key; non-text metadata fields
            # such as event types can otherwise appear as false transcripts.
            content = payload.get("content")
            if isinstance(content, list):
                for entry in content:
                    extracted = self._extract_text(entry)
                    if extracted:
                        return extracted
            return None

        if isinstance(payload, list):
            for item in payload:
                extracted = self._extract_text(item)
                if extracted:
                    return extracted
            return None

        return None
