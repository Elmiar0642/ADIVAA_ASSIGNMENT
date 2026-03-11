import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.logging import get_logger


class ConversationStore:
    def __init__(self, base_dir: str | None = None) -> None:
        resolved = base_dir or os.getenv("CONVERSATION_LOG_DIR", "data/conversations")
        self._base_dir = Path(resolved)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._logger = get_logger(__name__)

    async def append_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "role": role,
            "text": text,
            "metadata": metadata or {},
        }
        lock = await self._get_lock(session_id)
        async with lock:
            path = self._path_for_session(session_id)
            await asyncio.to_thread(self._append_jsonl, path, entry)
        self._logger.info(
            "conversation_turn_saved",
            extra={
                "session_id": session_id,
                "event_type": "conversation.turn.saved",
                "role": role,
                "path": str(path),
            },
        )

    async def append_user_turn(
        self,
        session_id: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.append_turn(session_id, "user", text, metadata=metadata)

    async def append_assistant_turn(
        self,
        session_id: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.append_turn(session_id, "assistant", text, metadata=metadata)

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            existing = self._locks.get(session_id)
            if existing is not None:
                return existing
            created = asyncio.Lock()
            self._locks[session_id] = created
            return created

    def _path_for_session(self, session_id: str) -> Path:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
        return self._base_dir / f"{safe or 'session'}.jsonl"

    def _append_jsonl(self, path: Path, entry: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True))
            handle.write("\n")
