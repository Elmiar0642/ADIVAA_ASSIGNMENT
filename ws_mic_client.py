#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import math
import signal
from typing import Any

import websockets

try:
    import simpleaudio as sa
except Exception:  # pragma: no cover - optional dependency
    sa = None


SAMPLE_RATE = 16000
FRAME_MS = 100
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # PCM16
DEFAULT_FREQUENCY_HZ = 440.0


class VoiceClient:
    def __init__(
        self,
        url: str,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = FRAME_MS,
        tone_hz: float = DEFAULT_FREQUENCY_HZ,
        volume: float = 0.20,
    ) -> None:
        self.url = url
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.tone_hz = tone_hz
        self.volume = volume
        self.samples_per_frame = int(sample_rate * (frame_ms / 1000.0))
        self._phase = 0.0
        self._running = True
        self._assistant_audio_buffer = bytearray()
        self._assistant_audio_sample_rate = sample_rate
        self._play_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        print(f"Connecting: {self.url}")
        async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
            print("Connected.")
            if sa is None:
                print("Audio playback disabled: install `simpleaudio` to hear assistant output.")

            send_task = asyncio.create_task(self._send_audio_loop(ws), name="send-audio")
            recv_task = asyncio.create_task(self._recv_loop(ws), name="recv-events")

            try:
                await asyncio.wait(
                    {send_task, recv_task},
                    return_when=asyncio.FIRST_EXCEPTION,
                )
            finally:
                self._running = False
                send_task.cancel()
                recv_task.cancel()
                await asyncio.gather(send_task, recv_task, return_exceptions=True)

    async def _send_audio_loop(self, ws: Any) -> None:
        interval = self.frame_ms / 1000.0
        while self._running:
            pcm = self._generate_pcm16_frame()
            payload = {
                "type": "audio_chunk",
                "format": "pcm16",
                "sample_rate": self.sample_rate,
                "data": base64.b64encode(pcm).decode("ascii"),
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(interval)

    async def _recv_loop(self, ws: Any) -> None:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue

            msg_type = message.get("type")
            payload = message.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}

            if msg_type in {"transcript.partial", "transcript.final", "transcript"}:
                text = payload.get("text", "")
                print(f"[{msg_type}] {text}")
                continue

            if msg_type in {"assistant_text", "assistant.text"}:
                text = payload.get("text", "")
                print(f"[assistant] {text}")
                continue

            if msg_type == "assistant_audio.chunk":
                chunk_b64 = payload.get("data")
                sample_rate = payload.get("sample_rate")
                if isinstance(sample_rate, int) and sample_rate > 0:
                    self._assistant_audio_sample_rate = sample_rate
                if isinstance(chunk_b64, str):
                    try:
                        self._assistant_audio_buffer.extend(base64.b64decode(chunk_b64))
                    except Exception:
                        pass
                continue

            if msg_type == "assistant_audio.completed":
                if self._assistant_audio_buffer:
                    await self._play_audio(
                        bytes(self._assistant_audio_buffer),
                        sample_rate=self._assistant_audio_sample_rate,
                    )
                    self._assistant_audio_buffer.clear()
                continue

            if msg_type == "error":
                print(f"[error] {payload}")

    def _generate_pcm16_frame(self) -> bytes:
        frame = bytearray(self.samples_per_frame * BYTES_PER_SAMPLE)
        amplitude = int(32767 * max(0.0, min(self.volume, 1.0)))
        phase_inc = 2.0 * math.pi * self.tone_hz / self.sample_rate

        for i in range(self.samples_per_frame):
            sample = int(amplitude * math.sin(self._phase))
            self._phase += phase_inc
            if self._phase >= 2.0 * math.pi:
                self._phase -= 2.0 * math.pi
            frame[2 * i : 2 * i + 2] = int(sample).to_bytes(2, byteorder="little", signed=True)
        return bytes(frame)

    async def _play_audio(self, pcm16_bytes: bytes, sample_rate: int) -> None:
        if not pcm16_bytes or sa is None:
            return
        if self._play_task and not self._play_task.done():
            self._play_task.cancel()
            await asyncio.gather(self._play_task, return_exceptions=True)
        self._play_task = asyncio.create_task(self._play_audio_blocking(pcm16_bytes, sample_rate))

    async def _play_audio_blocking(self, pcm16_bytes: bytes, sample_rate: int) -> None:
        def _play() -> None:
            play_obj = sa.play_buffer(
                pcm16_bytes,
                num_channels=CHANNELS,
                bytes_per_sample=BYTES_PER_SAMPLE,
                sample_rate=sample_rate,
            )
            play_obj.wait_done()

        await asyncio.to_thread(_play)

    def stop(self) -> None:
        self._running = False


async def _main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket microphone streaming simulator client")
    parser.add_argument("--url", default="ws://localhost:8000/ws/talk")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--frame-ms", type=int, default=FRAME_MS)
    parser.add_argument("--tone-hz", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--volume", type=float, default=0.20)
    args = parser.parse_args()

    client = VoiceClient(
        url=args.url,
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        tone_hz=args.tone_hz,
        volume=args.volume,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, client.stop)
        except NotImplementedError:
            # Windows event loop may not support add_signal_handler
            pass

    await client.run()


if __name__ == "__main__":
    asyncio.run(_main())
