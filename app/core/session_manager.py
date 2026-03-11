import asyncio
from typing import Any
from uuid import uuid4

from fastapi import WebSocket

from app.core.logging import get_logger
from app.core.metrics import metrics
from app.core.session import Session


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._logger = get_logger(__name__)

    async def create_session(
        self,
        websocket: WebSocket,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        resolved_id = session_id or str(uuid4())
        session = Session(
            session_id=resolved_id,
            websocket=websocket,
            metadata=metadata or {},
        )
        await session.start()

        async with self._lock:
            self._sessions[resolved_id] = session
            active_count = len(self._sessions)

        await metrics.increment_counter("ws_connections_total")
        await metrics.set_gauge("ws_active_sessions", active_count)
        self._logger.info(
            "session_created",
            extra={
                "session_id": resolved_id,
                "event_type": "session.created",
                "active_sessions": active_count,
            },
        )
        return session

    async def remove_session(self, session_id: str) -> None:
        async with self._lock:
            removed = self._sessions.pop(session_id, None)
            active_count = len(self._sessions)

        if removed is None:
            return

        await removed.close()
        await metrics.set_gauge("ws_active_sessions", active_count)
        self._logger.info(
            "session_removed",
            extra={
                "session_id": session_id,
                "event_type": "session.removed",
                "active_sessions": active_count,
            },
        )

    async def get_session(self, session_id: str) -> Session | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def list_session_ids(self) -> list[str]:
        async with self._lock:
            return list(self._sessions.keys())

    async def broadcast_json(
        self,
        payload: dict[str, Any],
        exclude_session_id: str | None = None,
    ) -> None:
        async with self._lock:
            targets = [
                session
                for sid, session in self._sessions.items()
                if sid != exclude_session_id
            ]

        if not targets:
            return

        results = await asyncio.gather(
            *(session.send_json(payload) for session in targets),
            return_exceptions=True,
        )

        failed_session_ids: list[str] = []
        for session, result in zip(targets, results, strict=False):
            if isinstance(result, Exception):
                failed_session_ids.append(session.session_id)
                self._logger.exception(
                    "broadcast_failed",
                    extra={
                        "session_id": session.session_id,
                        "event_type": "session.broadcast.failed",
                        "error": str(result),
                    },
                )

        for failed_session_id in failed_session_ids:
            await self.remove_session(failed_session_id)
