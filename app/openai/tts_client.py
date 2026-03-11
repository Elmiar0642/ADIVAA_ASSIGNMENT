import asyncio
import math
import os
from typing import AsyncIterator

import aiohttp

from app.core.logging import get_logger


class TTSClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "nova",
        base_url: str = "https://api.openai.com/v1",
        response_format: str = "pcm",
        timeout_seconds: float = 45.0,
        chunk_size_bytes: int = 4096,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = os.getenv("OPENAI_TTS_MODEL", model)
        self._voice = os.getenv("OPENAI_TTS_VOICE", voice)
        self._base_url = os.getenv("OPENAI_BASE_URL", base_url).rstrip("/")
        self._response_format = os.getenv("OPENAI_TTS_RESPONSE_FORMAT", response_format)
        self._output_sample_rate = int(os.getenv("OPENAI_TTS_SAMPLE_RATE", "24000"))
        self._timeout_seconds = timeout_seconds
        self._chunk_size_bytes = chunk_size_bytes
        self._fallback_sample_rate = 16000
        self._logger = get_logger(__name__)

    @property
    def output_sample_rate(self) -> int:
        return self._output_sample_rate

    async def stream_speech(self, text: str) -> AsyncIterator[bytes]:
        normalized = text.strip()
        if not normalized:
            return

        if not self._api_key:
            async for chunk in self._stream_fallback(normalized):
                yield chunk
            return

        endpoint = f"{self._base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "voice": self._voice,
            "input": normalized,
            "response_format": self._response_format,
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status >= 400:
                        body = await response.text()
                        self._logger.error(
                            "tts_request_failed",
                            extra={"status": response.status, "body": body[:1000]},
                        )
                        async for chunk in self._stream_fallback(normalized):
                            yield chunk
                        return

                    async for chunk in response.content.iter_chunked(self._chunk_size_bytes):
                        if chunk:
                            yield chunk
        except Exception as exc:
            self._logger.exception("tts_request_exception", extra={"error": str(exc)})
            async for chunk in self._stream_fallback(normalized):
                yield chunk

    async def synthesize(self, text: str) -> bytes:
        chunks: list[bytes] = []
        async for chunk in self.stream_speech(text):
            chunks.append(chunk)
        return b"".join(chunks)

    def _fallback_pcm16(self, text: str) -> bytes:
        chars = max(1, len(text))
        duration_seconds = max(0.35, min(2.5, chars * 0.025))
        total_samples = int(self._fallback_sample_rate * duration_seconds)
        frequency_hz = 440.0
        amplitude = int(32767 * 0.2)
        phase_increment = 2.0 * math.pi * frequency_hz / self._fallback_sample_rate

        frame = bytearray(total_samples * 2)
        phase = 0.0
        for index in range(total_samples):
            sample = int(amplitude * math.sin(phase))
            phase += phase_increment
            if phase >= 2.0 * math.pi:
                phase -= 2.0 * math.pi
            frame[2 * index : 2 * index + 2] = int(sample).to_bytes(
                2,
                byteorder="little",
                signed=True,
            )
        return bytes(frame)

    async def _stream_fallback(self, text: str) -> AsyncIterator[bytes]:
        payload = self._fallback_pcm16(text)
        for index in range(0, len(payload), self._chunk_size_bytes):
            yield payload[index : index + self._chunk_size_bytes]
            await asyncio.sleep(0)
