import asyncio
from collections import deque
import re
from typing import Any


def _normalize_metric_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_:]", "_", name).lower()


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    index = int((len(values) - 1) * q)
    return values[index]


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: dict[str, int] = {
            "ws_connections_total": 0,
            "ws_messages_in_total": 0,
            "ws_messages_out_total": 0,
            "ws_errors_total": 0,
        }
        self._gauges: dict[str, float] = {
            "ws_active_sessions": 0,
        }
        self._stage_latency_ms: dict[str, deque[float]] = {
            "stt": deque(maxlen=5000),
            "llm": deque(maxlen=5000),
            "tts": deque(maxlen=5000),
            "end_to_end": deque(maxlen=5000),
        }
        self._cost_per_turn_usd: deque[float] = deque(maxlen=5000)
        self._cost_per_conversation_usd: deque[float] = deque(maxlen=5000)
        self._conversation_cost_by_session: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def increment_counter(self, name: str, amount: int = 1) -> None:
        metric_name = _normalize_metric_name(name)
        async with self._lock:
            self._counters[metric_name] = self._counters.get(metric_name, 0) + amount

    async def set_gauge(self, name: str, value: int | float) -> None:
        metric_name = _normalize_metric_name(name)
        async with self._lock:
            self._gauges[metric_name] = float(value)

    async def snapshot(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            stage_stats = self._stage_snapshot()
            cost_stats = self._cost_snapshot()
            return {
                "active_sessions": int(self._gauges.get("ws_active_sessions", 0)),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latency_ms": stage_stats,
                "cost_usd": cost_stats,
            }

    async def observe_stage_latency(self, stage: str, latency_ms: float) -> None:
        if latency_ms < 0:
            return
        normalized_stage = _normalize_metric_name(stage)
        async with self._lock:
            bucket = self._stage_latency_ms.setdefault(normalized_stage, deque(maxlen=5000))
            bucket.append(float(latency_ms))

    async def observe_end_to_end_latency(self, latency_ms: float) -> None:
        await self.observe_stage_latency("end_to_end", latency_ms)

    async def observe_turn_cost(self, cost_usd: float) -> None:
        if cost_usd < 0:
            return
        async with self._lock:
            self._cost_per_turn_usd.append(float(cost_usd))

    async def observe_conversation_cost(self, session_id: str, cost_usd: float) -> None:
        if cost_usd < 0:
            return
        async with self._lock:
            self._cost_per_conversation_usd.append(float(cost_usd))
            self._conversation_cost_by_session[session_id] = float(cost_usd)

    def _stage_snapshot(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for stage_name, bucket in self._stage_latency_ms.items():
            values = sorted(bucket)
            if not values:
                output[stage_name] = {
                    "count": 0,
                    "avg": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "max": 0.0,
                }
                continue

            count = len(values)
            output[stage_name] = {
                "count": count,
                "avg": round(sum(values) / count, 3),
                "p50": round(_quantile(values, 0.50), 3),
                "p95": round(_quantile(values, 0.95), 3),
                "max": round(values[-1], 3),
            }
        return output

    def _cost_snapshot(self) -> dict[str, Any]:
        turn_values = sorted(self._cost_per_turn_usd)
        convo_values = sorted(self._cost_per_conversation_usd)

        def stats(values: list[float]) -> dict[str, float | int]:
            if not values:
                return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
            count = len(values)
            return {
                "count": count,
                "avg": round(sum(values) / count, 6),
                "p50": round(_quantile(values, 0.50), 6),
                "p95": round(_quantile(values, 0.95), 6),
                "max": round(values[-1], 6),
            }

        return {
            "per_turn": stats(turn_values),
            "per_conversation": stats(convo_values),
            "by_session": dict(self._conversation_cost_by_session),
        }


metrics = MetricsRegistry()
