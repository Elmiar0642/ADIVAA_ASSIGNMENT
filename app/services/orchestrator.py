from typing import Any

from app.core.logging import get_logger
from app.core.metrics import metrics
from app.core.session import Session
from app.models.events import ClientEvent, EventType, ServerEvent, parse_client_event
from app.openai.llm_client import LLMClient
from app.openai.realtime_client import RealtimeClient
from app.openai.stt_client import STTClient
from app.openai.tts_client import TTSClient
from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.types import ToolContext


class Orchestrator:
    def __init__(
        self,
        realtime_client: RealtimeClient | None = None,
        stt_client: STTClient | None = None,
        llm_client: LLMClient | None = None,
        tts_client: TTSClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.realtime_client = realtime_client or RealtimeClient()
        self.stt_client = stt_client or STTClient()
        self.llm_client = llm_client or LLMClient()
        self.tts_client = tts_client or TTSClient()
        self.tool_registry = tool_registry or build_default_registry()
        self._logger = get_logger(__name__)

    async def on_session_start(self, session: Session) -> None:
        await session.send_json(
            ServerEvent(
                type=EventType.SESSION_STARTED.value,
                session_id=session.session_id,
                payload={"message": "session_connected"},
            ).model_dump(mode="json")
        )
        await metrics.increment_counter("ws_messages_out_total")
        self._logger.info("session_started", extra={"session_id": session.session_id})

    async def on_session_end(self, session: Session) -> None:
        self._logger.info("session_ended", extra={"session_id": session.session_id})

    async def handle_text_message(self, session: Session, raw_text: str) -> None:
        event = parse_client_event(raw_text)

        if event.type == EventType.ERROR.value:
            await self._send_error(session, "invalid_event_payload", event.payload)
            return
        if event.type == EventType.PING.value:
            await self._send_event(
                session,
                EventType.PONG.value,
                {"message": "pong"},
            )
            return
        if event.type == EventType.TOOL_CALL.value:
            await self._handle_tool_call(session, event)
            return

        text_input = event.payload.get("text", raw_text)
        if not isinstance(text_input, str):
            text_input = str(text_input)

        assistant_text = await self.llm_client.generate_response(text_input)
        await self._send_event(
            session,
            EventType.ASSISTANT_TEXT.value,
            {"text": assistant_text},
        )

    async def handle_audio_chunk(
        self,
        session: Session,
        audio_chunk: bytes,
        audio_format: str = "pcm16",
        sample_rate: int = 16000,
    ) -> None:
        await session.enqueue_audio(audio_chunk, audio_format=audio_format, sample_rate=sample_rate)

    async def handle_control_message(
        self,
        session: Session,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        return await session.handle_control(action=action, payload=payload)

    async def _handle_tool_call(self, session: Session, event: ClientEvent) -> None:
        payload = event.payload
        tool_name = payload.get("name")
        tool_args = payload.get("arguments", {})

        if not isinstance(tool_name, str) or not tool_name:
            await self._send_error(session, "tool_name_missing", {"payload": payload})
            return
        if not isinstance(tool_args, dict):
            await self._send_error(session, "tool_args_invalid", {"payload": payload})
            return

        result = await self.tool_registry.call(
            tool_name,
            tool_args,
            context=ToolContext(
                session_id=session.session_id,
                emit_event=session.emit_event,
            ),
        )
        await self._send_event(
            session,
            EventType.TOOL_RESULT.value,
            {"name": tool_name, "result": result},
        )

    async def _send_event(
        self,
        session: Session,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        await session.send_json(
            ServerEvent(
                type=event_type,
                session_id=session.session_id,
                payload=payload,
            ).model_dump(mode="json")
        )
        await metrics.increment_counter("ws_messages_out_total")

    async def _send_error(
        self,
        session: Session,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        await metrics.increment_counter("ws_errors_total")
        await self._send_event(
            session,
            EventType.ERROR.value,
            {
                "message": message,
                "details": details or {},
            },
        )
