import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import time
from typing import Any
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from app.core.logging import get_logger
from app.core.metrics import metrics
from app.core.conversation_store import ConversationStore
from app.openai.llm_client import LLMClient
from app.openai.realtime_client import RealtimeClient
from app.openai.tts_client import TTSClient
from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.types import ToolContext


@dataclass(slots=True)
class AudioFrame:
    audio: bytes
    audio_format: str
    sample_rate: int
    received_at: float = field(default_factory=time.perf_counter)
    received_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(slots=True)
class TranscriptItem:
    text: str
    source: str
    is_final: bool
    received_at: float
    received_at_utc: str
    audio_frame_at: float | None = None
    audio_frame_at_utc: str | None = None
    latency_ms: float | None = None


@dataclass(slots=True)
class LLMItem:
    prompt: str
    source: str
    received_at: float
    turn_id: str | None = None
    turn_started_at: float | None = None
    stt_latency_ms: float | None = None
    estimated_cost_usd: float = 0.0


@dataclass(slots=True)
class ToolInvocation:
    name: str
    arguments: dict[str, Any]
    received_at: float = field(default_factory=time.perf_counter)
    turn_id: str | None = None
    turn_started_at: float | None = None
    estimated_cost_usd: float = 0.0
    llm_latency_ms: float | None = None


@dataclass(slots=True)
class TTSItem:
    text: str
    source: str
    received_at: float
    turn_id: str | None = None
    turn_started_at: float | None = None
    estimated_cost_usd: float = 0.0
    stt_latency_ms: float | None = None
    llm_latency_ms: float | None = None


@dataclass(slots=True)
class ConversationState:
    turns: list[dict[str, Any]] = field(default_factory=list)
    last_user_transcript: str | None = None
    active: bool = True


@dataclass(slots=True)
class SessionPipelineMetrics:
    audio_frames_received: int = 0
    transcripts_generated: int = 0
    llm_requests: int = 0
    tool_calls: int = 0
    tts_responses: int = 0
    tts_time_to_first_audio_samples: int = 0
    total_stt_latency_ms: float = 0.0
    total_llm_latency_ms: float = 0.0
    total_tts_latency_ms: float = 0.0
    total_tts_time_to_first_audio_ms: float = 0.0
    estimated_cost_usd: float = 0.0

    def snapshot(self) -> dict[str, float | int]:
        stt_avg = self.total_stt_latency_ms / self.transcripts_generated if self.transcripts_generated else 0.0
        llm_avg = self.total_llm_latency_ms / self.llm_requests if self.llm_requests else 0.0
        tts_avg = self.total_tts_latency_ms / self.tts_responses if self.tts_responses else 0.0
        tts_ttfb_avg = (
            self.total_tts_time_to_first_audio_ms / self.tts_time_to_first_audio_samples
            if self.tts_time_to_first_audio_samples
            else 0.0
        )
        return {
            "audio_frames_received": self.audio_frames_received,
            "transcripts_generated": self.transcripts_generated,
            "llm_requests": self.llm_requests,
            "tool_calls": self.tool_calls,
            "tts_responses": self.tts_responses,
            "avg_stt_latency_ms": round(stt_avg, 3),
            "avg_llm_latency_ms": round(llm_avg, 3),
            "avg_tts_latency_ms": round(tts_avg, 3),
            "avg_tts_time_to_first_audio_ms": round(tts_ttfb_avg, 3),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


@dataclass(slots=True)
class Session:
    session_id: str
    websocket: WebSocket
    metadata: dict[str, Any] = field(default_factory=dict)
    realtime_client: RealtimeClient = field(default_factory=RealtimeClient)
    llm_client: LLMClient = field(default_factory=LLMClient)
    tts_client: TTSClient = field(default_factory=TTSClient)
    tool_registry: ToolRegistry = field(default_factory=build_default_registry)
    conversation_store: ConversationStore = field(default_factory=ConversationStore)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_state: ConversationState = field(default_factory=ConversationState)
    pipeline_metrics: SessionPipelineMetrics = field(default_factory=SessionPipelineMetrics)
    audio_input_queue: asyncio.Queue[AudioFrame | None] = field(default_factory=asyncio.Queue)
    transcript_queue: asyncio.Queue[TranscriptItem | None] = field(default_factory=asyncio.Queue)
    llm_queue: asyncio.Queue[LLMItem | None] = field(default_factory=asyncio.Queue)
    tts_queue: asyncio.Queue[TTSItem | None] = field(default_factory=asyncio.Queue)
    _tool_queue: asyncio.Queue[ToolInvocation | None] = field(default_factory=asyncio.Queue, repr=False)
    _tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict, repr=False)
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _teardown_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _logger: Any = field(init=False, repr=False)
    _last_audio_frame_at: float | None = field(default=None, init=False, repr=False)
    _last_audio_frame_at_utc: str | None = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._logger = get_logger(__name__)

    def _log_event(
        self,
        event_type: str,
        *,
        latency_ms: float | None = None,
        cost_usd: float | None = None,
        **fields: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "event_type": event_type,
        }
        if latency_ms is not None:
            payload["latency_ms"] = round(latency_ms, 3)
        if cost_usd is not None:
            payload["cost_usd"] = round(cost_usd, 6)
        payload.update(fields)
        self._logger.info(event_type, extra=payload)

    @property
    def is_connected(self) -> bool:
        return self.websocket.client_state == WebSocketState.CONNECTED

    @property
    def is_running(self) -> bool:
        return self._running and not self._closed

    async def start(self) -> None:
        if self._running or self._closed:
            return

        self.realtime_client.set_callbacks(
            on_transcript=self._on_realtime_transcript,
            on_error=self._on_realtime_error,
        )
        await self.realtime_client.connect()
        self._running = True
        self.conversation_state.active = True

        self._spawn_stage("audio_ingestion", self._audio_ingestion_worker)
        self._spawn_stage("transcription", self._transcription_worker)
        self._spawn_stage("llm_reasoning", self._llm_reasoning_worker)
        self._spawn_stage("tool_execution", self._tool_execution_worker)
        self._spawn_stage("tts_streaming", self._tts_streaming_worker)

        self._logger.info(
            "session_pipeline_started",
            extra={
                "session_id": self.session_id,
                "event_type": "session.pipeline.started",
                "stages": sorted(self._tasks.keys()),
            },
        )

    async def send_json(self, payload: dict[str, Any]) -> bool:
        async with self._send_lock:
            if not self.is_connected:
                return False
            try:
                await self.websocket.send_json(payload)
                return True
            except Exception:
                await metrics.increment_counter("ws_errors_total")
                self._logger.exception(
                    "session_send_failed",
                    extra={"session_id": self.session_id, "event_type": "session.send.failed"},
                )
                return False

    async def emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        sent = await self.send_json(
            {
                "type": event_type,
                "session_id": self.session_id,
                "payload": payload,
            }
        )
        if sent:
            await metrics.increment_counter("ws_messages_out_total")

    async def enqueue_audio(
        self,
        audio: bytes,
        audio_format: str = "pcm16",
        sample_rate: int = 16000,
    ) -> None:
        if self._closed:
            return

        frame = AudioFrame(audio=audio, audio_format=audio_format, sample_rate=sample_rate)
        await self.audio_input_queue.put(frame)
        self.pipeline_metrics.audio_frames_received += 1
        self.pipeline_metrics.estimated_cost_usd += self._estimate_stt_cost(frame)
        self._log_event(
            "session.audio.enqueued",
            queue_size=self.audio_input_queue.qsize(),
            audio_bytes=len(audio),
            audio_format=audio_format,
            sample_rate=sample_rate,
        )

    async def enqueue_text(self, text: str, source: str = "client") -> None:
        if self._closed:
            return
        await self.llm_queue.put(LLMItem(prompt=text, source=source, received_at=time.perf_counter()))

    async def handle_control(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        normalized_action = action.strip().lower()
        if normalized_action == "ping":
            await self.emit_event("pong", {"action": "pong"})
            return False

        if normalized_action == "flush":
            self._flush_queues()
            await self.emit_event("control_ack", {"action": "flush", "status": "ok"})
            return False

        if normalized_action == "text":
            text_payload = (payload or {}).get("text")
            if isinstance(text_payload, str):
                await self.transcript_queue.put(
                    TranscriptItem(
                        text=text_payload,
                        source="control",
                        is_final=True,
                        received_at=time.perf_counter(),
                        received_at_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )
                await self.emit_event("control_ack", {"action": "text", "status": "queued"})
            else:
                await self.emit_event("error", {"message": "control.text requires payload.text"})
            return False

        if normalized_action in {"disconnect", "close", "stop"}:
            await self.emit_event(
                "control_ack",
                {"action": normalized_action, "status": "closing"},
            )
            return True

        await self.emit_event(
            "control_ack",
            {"action": normalized_action or "unknown", "status": "ok"},
        )
        return False

    def metrics_snapshot(self) -> dict[str, Any]:
        snapshot = self.pipeline_metrics.snapshot()
        snapshot.update(
            {
                "session_id": self.session_id,
                "active": self.conversation_state.active,
                "turns": len(self.conversation_state.turns),
            }
        )
        return snapshot

    async def teardown(self, reason: str = "client_disconnect") -> None:
        async with self._teardown_lock:
            if self._closed:
                return

            self._closed = True
            self._running = False
            self.conversation_state.active = False

            self._push_shutdown_sentinels()
            await self._stop_all_tasks()
            await self.realtime_client.disconnect()
            await metrics.observe_conversation_cost(self.session_id, self.pipeline_metrics.estimated_cost_usd)

            self._logger.info(
                "session_teardown_complete",
                extra={
                    "session_id": self.session_id,
                    "event_type": "session.teardown.completed",
                    "reason": reason,
                    "metrics": self.metrics_snapshot(),
                    "cost_usd": round(self.pipeline_metrics.estimated_cost_usd, 6),
                },
            )

    async def close(self, code: int = 1000) -> None:
        await self.teardown(reason="close")
        if self.is_connected:
            await self.websocket.close(code=code)

    def _spawn_stage(self, stage_name: str, worker: Any) -> None:
        task = asyncio.create_task(self._run_stage(stage_name, worker), name=f"{self.session_id}:{stage_name}")
        self._tasks[stage_name] = task

    async def _run_stage(self, stage_name: str, worker: Any) -> None:
        try:
            await worker()
        except asyncio.CancelledError:
            self._logger.info(
                "session_stage_cancelled",
                extra={
                    "session_id": self.session_id,
                    "event_type": "session.stage.cancelled",
                    "stage": stage_name,
                },
            )
            raise
        except Exception:
            await metrics.increment_counter("ws_errors_total")
            self._logger.exception(
                "session_stage_failed",
                extra={
                    "session_id": self.session_id,
                    "event_type": "session.stage.failed",
                    "stage": stage_name,
                },
            )

    async def _audio_ingestion_worker(self) -> None:
        while True:
            frame = await self.audio_input_queue.get()
            try:
                if frame is None:
                    return

                self._last_audio_frame_at = frame.received_at
                self._last_audio_frame_at_utc = frame.received_at_utc
                await self.realtime_client.send_audio_frame(
                    frame.audio,
                    audio_format=frame.audio_format,
                    sample_rate=frame.sample_rate,
                )
            finally:
                self.audio_input_queue.task_done()

    async def _transcription_worker(self) -> None:
        while True:
            transcript_item = await self.transcript_queue.get()
            try:
                if transcript_item is None:
                    return

                if not transcript_item.is_final:
                    continue

                transcript = transcript_item.text
                if not transcript:
                    continue

                turn_id = str(uuid4())
                turn_started_at = transcript_item.audio_frame_at or transcript_item.received_at
                stt_cost = self._estimate_stt_turn_cost(transcript)
                self.pipeline_metrics.transcripts_generated += 1
                if transcript_item.latency_ms is not None:
                    self.pipeline_metrics.total_stt_latency_ms += transcript_item.latency_ms
                    await metrics.observe_stage_latency("stt", transcript_item.latency_ms)
                    self._log_event(
                        "stt.final_transcript",
                        latency_ms=transcript_item.latency_ms,
                        source=transcript_item.source,
                        turn_id=turn_id,
                    )
                self.conversation_state.last_user_transcript = transcript
                self.conversation_state.turns.append(
                    {
                        "role": "user",
                        "text": transcript,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                await self.conversation_store.append_user_turn(
                    self.session_id,
                    transcript,
                    metadata={
                        "source": transcript_item.source,
                        "turn_id": turn_id,
                        "latency_ms": transcript_item.latency_ms,
                    },
                )
                await self.emit_event(
                    "transcript.final",
                    {
                        "text": transcript,
                        "source": transcript_item.source,
                        "audio_frame_at_utc": transcript_item.audio_frame_at_utc,
                        "transcript_at_utc": transcript_item.received_at_utc,
                        "latency_ms": transcript_item.latency_ms,
                    },
                )
                await self.llm_queue.put(
                    LLMItem(
                        prompt=transcript,
                        source=transcript_item.source,
                        received_at=time.perf_counter(),
                        turn_id=turn_id,
                        turn_started_at=turn_started_at,
                        stt_latency_ms=transcript_item.latency_ms,
                        estimated_cost_usd=stt_cost,
                    )
                )
            finally:
                self.transcript_queue.task_done()

    async def _llm_reasoning_worker(self) -> None:
        while True:
            item = await self.llm_queue.get()
            try:
                if item is None:
                    return

                started = time.perf_counter()
                llm_result = await self.llm_client.reason(item.prompt)
                latency_ms = (time.perf_counter() - started) * 1000
                self.pipeline_metrics.llm_requests += 1
                self.pipeline_metrics.total_llm_latency_ms += latency_ms
                await metrics.observe_stage_latency("llm", latency_ms)

                if llm_result.tool_call is not None:
                    tool_call = llm_result.tool_call
                    llm_cost = self._estimate_llm_cost(item.prompt, json.dumps(tool_call.arguments, default=str))
                    tool_payload = {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                    self.conversation_state.turns.append(
                        {
                            "role": "assistant",
                            "tool_call": tool_payload,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    await self.emit_event("tool_call", tool_payload)
                    await self._tool_queue.put(
                        ToolInvocation(
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            turn_id=item.turn_id,
                            turn_started_at=item.turn_started_at,
                            estimated_cost_usd=item.estimated_cost_usd + llm_cost,
                            llm_latency_ms=latency_ms,
                        )
                    )
                    self.pipeline_metrics.estimated_cost_usd += llm_cost
                    self._log_event(
                        "llm.tool_call",
                        latency_ms=latency_ms,
                        cost_usd=llm_cost,
                        tool_name=tool_call.name,
                        turn_id=item.turn_id,
                    )
                    continue

                assistant_text = (llm_result.text or "").strip()
                if not assistant_text:
                    continue

                llm_cost = self._estimate_llm_cost(item.prompt, assistant_text)
                self.pipeline_metrics.estimated_cost_usd += llm_cost
                self._log_event(
                    "llm.assistant_text",
                    latency_ms=latency_ms,
                    cost_usd=llm_cost,
                    turn_id=item.turn_id,
                )
                self.conversation_state.turns.append(
                    {
                        "role": "assistant",
                        "text": assistant_text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                await self.conversation_store.append_assistant_turn(
                    self.session_id,
                    assistant_text,
                    metadata={
                        "source": "llm",
                        "turn_id": item.turn_id,
                        "latency_ms": latency_ms,
                    },
                )
                await self.emit_event("assistant_text", {"text": assistant_text})
                await self.tts_queue.put(
                    TTSItem(
                        text=assistant_text,
                        source="llm",
                        received_at=time.perf_counter(),
                        turn_id=item.turn_id,
                        turn_started_at=item.turn_started_at,
                        estimated_cost_usd=item.estimated_cost_usd + llm_cost,
                        stt_latency_ms=item.stt_latency_ms,
                        llm_latency_ms=latency_ms,
                    )
                )
            finally:
                self.llm_queue.task_done()

    async def _tool_execution_worker(self) -> None:
        while True:
            invocation = await self._tool_queue.get()
            try:
                if invocation is None:
                    return

                self.pipeline_metrics.tool_calls += 1
                started = time.perf_counter()
                result = await self.tool_registry.call(
                    invocation.name,
                    invocation.arguments,
                    context=ToolContext(
                        session_id=self.session_id,
                        emit_event=self.emit_event,
                    ),
                )
                latency_ms = (time.perf_counter() - started) * 1000
                self._log_event(
                    "tool.executed",
                    latency_ms=latency_ms,
                    tool_name=invocation.name,
                    turn_id=invocation.turn_id,
                )
                await self.emit_event(
                    "tool_result",
                    {
                        "name": invocation.name,
                        "arguments": invocation.arguments,
                        "result": result,
                        "latency_ms": round(latency_ms, 3),
                    },
                )
            finally:
                self._tool_queue.task_done()

    async def _tts_streaming_worker(self) -> None:
        while True:
            item = await self.tts_queue.get()
            try:
                if item is None:
                    return

                started = time.perf_counter()
                first_audio_at: float | None = None
                total_audio_bytes = 0
                chunk_index = 0

                async for chunk in self.tts_client.stream_speech(item.text):
                    now = time.perf_counter()
                    if first_audio_at is None:
                        first_audio_at = now

                    total_audio_bytes += len(chunk)
                    await self.emit_event(
                        "assistant_audio.chunk",
                        {
                            "chunk_index": chunk_index,
                            "format": "pcm16",
                            "sample_rate": self.tts_client.output_sample_rate,
                            "data": base64.b64encode(chunk).decode("ascii"),
                            "text": item.text if chunk_index == 0 else None,
                        },
                    )
                    chunk_index += 1

                total_tts_latency_ms = (time.perf_counter() - started) * 1000
                time_to_first_audio_ms = (
                    (first_audio_at - started) * 1000 if first_audio_at is not None else None
                )

                self.pipeline_metrics.tts_responses += 1
                self.pipeline_metrics.total_tts_latency_ms += total_tts_latency_ms
                await metrics.observe_stage_latency("tts", total_tts_latency_ms)
                if time_to_first_audio_ms is not None:
                    self.pipeline_metrics.tts_time_to_first_audio_samples += 1
                    self.pipeline_metrics.total_tts_time_to_first_audio_ms += time_to_first_audio_ms
                tts_cost = self._estimate_tts_cost(item.text)
                self.pipeline_metrics.estimated_cost_usd += tts_cost

                end_to_end_latency_ms: float | None = None
                if item.turn_started_at is not None:
                    end_to_end_latency_ms = (time.perf_counter() - item.turn_started_at) * 1000
                    await metrics.observe_end_to_end_latency(end_to_end_latency_ms)

                turn_cost_usd = item.estimated_cost_usd + tts_cost
                await metrics.observe_turn_cost(turn_cost_usd)
                self._log_event(
                    "tts.completed",
                    latency_ms=total_tts_latency_ms,
                    cost_usd=tts_cost,
                    turn_id=item.turn_id,
                    time_to_first_audio_ms=round(time_to_first_audio_ms, 3)
                    if time_to_first_audio_ms is not None
                    else None,
                    end_to_end_latency_ms=round(end_to_end_latency_ms, 3)
                    if end_to_end_latency_ms is not None
                    else None,
                    estimated_cost_per_turn_usd=round(turn_cost_usd, 6),
                )

                await self.emit_event(
                    "assistant_audio.completed",
                    {
                        "chunks_sent": chunk_index,
                        "total_audio_bytes": total_audio_bytes,
                        "time_to_first_audio_ms": round(time_to_first_audio_ms, 3)
                        if time_to_first_audio_ms is not None
                        else None,
                        "total_tts_latency_ms": round(total_tts_latency_ms, 3),
                        "end_to_end_latency_ms": round(end_to_end_latency_ms, 3)
                        if end_to_end_latency_ms is not None
                        else None,
                        "estimated_cost_per_turn_usd": round(turn_cost_usd, 6),
                    },
                )
            finally:
                self.tts_queue.task_done()

    def _flush_queues(self) -> None:
        self._drain_queue(self.audio_input_queue)
        self._drain_queue(self.transcript_queue)
        self._drain_queue(self.llm_queue)
        self._drain_queue(self.tts_queue)
        self._drain_queue(self._tool_queue)

    def _drain_queue(self, queue: asyncio.Queue[Any]) -> None:
        while True:
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                return

    def _push_shutdown_sentinels(self) -> None:
        self.audio_input_queue.put_nowait(None)
        self.transcript_queue.put_nowait(None)
        self.llm_queue.put_nowait(None)
        self.tts_queue.put_nowait(None)
        self._tool_queue.put_nowait(None)

    async def _on_realtime_transcript(
        self,
        text: str,
        is_final: bool,
        event: dict[str, Any],
    ) -> None:
        if self._closed:
            return

        event_type = str(event.get("type", ""))

        now = time.perf_counter()
        now_utc = datetime.now(timezone.utc).isoformat()
        audio_at = self._last_audio_frame_at
        latency_ms = ((now - audio_at) * 1000) if audio_at is not None else None

        transcript_item = TranscriptItem(
            text=text,
            source="realtime",
            is_final=is_final,
            received_at=now,
            received_at_utc=now_utc,
            audio_frame_at=audio_at,
            audio_frame_at_utc=self._last_audio_frame_at_utc,
            latency_ms=latency_ms,
        )
        await self.transcript_queue.put(transcript_item)
        self._log_event(
            "stt.partial_transcript" if not is_final else "stt.final_event_received",
            latency_ms=latency_ms,
            source="realtime",
        )

        if not is_final:
            await self.emit_event(
                "transcript.partial",
                {
                    "text": text,
                    "source": "realtime",
                    "audio_frame_at_utc": transcript_item.audio_frame_at_utc,
                    "transcript_at_utc": transcript_item.received_at_utc,
                    "latency_ms": transcript_item.latency_ms,
                    "event_type": event_type,
                },
            )

    async def _on_realtime_error(
        self,
        message: str,
        details: dict[str, Any] | None,
    ) -> None:
        await self.emit_event(
            "error",
            {
                "message": message,
                "details": details or {},
                "source": "openai_realtime",
            },
        )

    async def _stop_all_tasks(self) -> None:
        if not self._tasks:
            return

        running_tasks = list(self._tasks.values())
        done, pending = await asyncio.wait(running_tasks, timeout=2.0)

        for pending_task in pending:
            pending_task.cancel()

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if done:
            await asyncio.gather(*done, return_exceptions=True)

        self._tasks.clear()

    def _estimate_stt_cost(self, frame: AudioFrame) -> float:
        if frame.sample_rate <= 0:
            return 0.0
        # Rough placeholder: 16-bit mono -> 2 bytes/sample.
        audio_seconds = len(frame.audio) / (frame.sample_rate * 2)
        return audio_seconds * 0.0001

    def _estimate_stt_turn_cost(self, transcript: str) -> float:
        # Placeholder estimate by transcript length when exact audio attribution is unavailable.
        estimated_tokens = max(1, len(transcript) // 4)
        return estimated_tokens * 0.000001

    def _estimate_llm_cost(self, prompt: str, response: str) -> float:
        estimated_tokens = max(1, (len(prompt) + len(response)) // 4)
        return estimated_tokens * 0.000002

    def _estimate_tts_cost(self, text: str) -> float:
        return max(1, len(text)) * 0.0000005
