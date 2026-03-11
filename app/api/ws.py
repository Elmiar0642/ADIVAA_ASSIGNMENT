import base64
import binascii

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.core.logging import get_logger
from app.core.metrics import metrics
from app.core.session import Session
from app.core.session_manager import SessionManager
from app.models.events import (
    AudioChunkMessage,
    ControlMessage,
    EventType,
    ServerEvent,
    parse_ws_message,
)
from app.services.orchestrator import Orchestrator

router = APIRouter(tags=["talk"])
logger = get_logger(__name__)
session_manager = SessionManager()
orchestrator = Orchestrator()


@router.websocket("/ws/talk")
async def talk_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session: Session | None = None

    try:
        metadata = {"client_host": websocket.client.host if websocket.client else None}
        session = await session_manager.create_session(websocket=websocket, metadata=metadata)
        logger.info(
            "ws_connected",
            extra={
                "session_id": session.session_id,
                "event_type": "ws.connected",
                "client_host": metadata.get("client_host"),
            },
        )
        await orchestrator.on_session_start(session)

        while True:
            message = await websocket.receive()
            await metrics.increment_counter("ws_messages_in_total")
            message_type = message.get("type")

            if message_type == "websocket.disconnect":
                logger.info(
                    "ws_disconnect_event",
                    extra={"session_id": session.session_id, "event_type": "ws.disconnect.event"},
                )
                break
            if message_type != "websocket.receive":
                continue

            text_payload = message.get("text")
            bytes_payload = message.get("bytes")
            if bytes_payload is not None:
                await orchestrator.handle_audio_chunk(
                    session,
                    bytes_payload,
                    audio_format="pcm16",
                    sample_rate=16000,
                )
                continue

            if text_payload is None:
                await _send_ws_error(
                    session,
                    "empty_message",
                    {"detail": "no text or bytes payload found"},
                )
                continue

            try:
                incoming_message = parse_ws_message(text_payload)
            except ValueError as exc:
                await _send_ws_error(
                    session,
                    "invalid_message",
                    {"detail": str(exc)},
                )
                continue

            if isinstance(incoming_message, AudioChunkMessage):
                try:
                    audio_bytes = base64.b64decode(incoming_message.data, validate=True)
                except (binascii.Error, ValueError):
                    await _send_ws_error(
                        session,
                        "invalid_audio_base64",
                        {"detail": "audio_chunk.data must be valid base64"},
                    )
                    continue

                await orchestrator.handle_audio_chunk(
                    session,
                    audio_bytes,
                    audio_format=incoming_message.format,
                    sample_rate=incoming_message.sample_rate,
                )
                continue

            if isinstance(incoming_message, ControlMessage):
                should_close = await orchestrator.handle_control_message(
                    session,
                    action=incoming_message.action,
                    payload=incoming_message.payload,
                )
                if should_close:
                    break
    except WebSocketDisconnect:
        if session is not None:
            logger.info(
                "ws_disconnected",
                extra={"session_id": session.session_id, "event_type": "ws.disconnected"},
            )
    except Exception:
        if session is not None:
            logger.exception(
                "ws_error",
                extra={"session_id": session.session_id, "event_type": "ws.error"},
            )
            if websocket.client_state == WebSocketState.CONNECTED:
                await _send_ws_error(session, "internal_server_error")
    finally:
        if session is not None:
            await orchestrator.on_session_end(session)
            await session_manager.remove_session(session.session_id)
            logger.info(
                "ws_cleanup_complete",
                extra={"session_id": session.session_id, "event_type": "ws.cleanup.complete"},
            )


async def _send_ws_error(
    session: Session,
    message: str,
    details: dict[str, str] | None = None,
) -> None:
    await metrics.increment_counter("ws_errors_total")
    await session.send_json(
        ServerEvent(
            type=EventType.ERROR.value,
            session_id=session.session_id,
            payload={
                "message": message,
                "details": details or {},
            },
        ).model_dump(mode="json")
    )
    await metrics.increment_counter("ws_messages_out_total")
