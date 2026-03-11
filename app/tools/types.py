from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(slots=True)
class ToolContext:
    session_id: str
    emit_event: Callable[[str, dict[str, Any]], Awaitable[None]]
