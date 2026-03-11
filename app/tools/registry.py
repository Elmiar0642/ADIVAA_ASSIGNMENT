import inspect
from typing import Any, Awaitable, Callable

from app.tools.play_audio import play_audio_tool
from app.tools.types import ToolContext

ToolResult = dict[str, Any]


ToolFunction = Callable[[dict[str, Any], ToolContext], ToolResult | Awaitable[ToolResult]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}

    def register(self, name: str, fn: ToolFunction) -> None:
        self._tools[name] = fn

    async def call(self, name: str, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown_tool:{name}"}

        result = tool(arguments, context)
        if inspect.isawaitable(result):
            return await result
        return result

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("play_audio", play_audio_tool)
    return registry
