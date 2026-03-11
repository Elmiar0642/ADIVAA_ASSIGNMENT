import base64
import contextlib
import os
from pathlib import Path
import time
from typing import Any
import wave

from app.core.logging import get_logger
from app.tools.types import ToolContext

_logger = get_logger(__name__)
_DEFAULT_CHUNK_BYTES = 4096


def _resolve_assets_dir(explicit_assets_dir: str | None) -> Path | None:
    if isinstance(explicit_assets_dir, str) and explicit_assets_dir.strip():
        return Path(explicit_assets_dir.strip())
    env_dir = os.getenv("PLAY_AUDIO_ASSETS_DIR")
    if isinstance(env_dir, str) and env_dir.strip():
        return Path(env_dir.strip())
    fallback = Path.cwd() / "assets"
    if fallback.exists() and fallback.is_dir():
        return fallback
    return None


def _resolve_track_file(track: str, assets_dir: Path | None) -> Path | None:
    if assets_dir is None:
        return None
    candidate = assets_dir / track
    resolved = _resolve_case_insensitive_file(candidate)
    if resolved is not None:
        return resolved

    if "." not in track:
        for suffix in (".wav", ".pcm", ".mp3", ".ogg"):
            alt = assets_dir / f"{track}{suffix}"
            resolved_alt = _resolve_case_insensitive_file(alt)
            if resolved_alt is not None:
                return resolved_alt
    return None


def _read_wav_pcm16(path: Path) -> tuple[bytes, int]:
    with contextlib.closing(wave.open(str(path), "rb")) as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())

    pcm16 = _to_pcm16_mono(raw, sample_width=sample_width, channels=channels)
    return pcm16, sample_rate


def _to_pcm16_mono(raw: bytes, *, sample_width: int, channels: int) -> bytes:
    if channels <= 0 or sample_width <= 0:
        return b""
    frame_bytes = sample_width * channels
    if frame_bytes <= 0:
        return b""
    frame_count = len(raw) // frame_bytes
    if frame_count <= 0:
        return b""

    out = bytearray(frame_count * 2)

    def decode_sample(chunk: bytes) -> int:
        if sample_width == 1:
            # 8-bit PCM WAV is unsigned.
            return (chunk[0] - 128) << 8
        if sample_width == 2:
            return int.from_bytes(chunk, byteorder="little", signed=True)
        if sample_width == 3:
            value = int.from_bytes(chunk + (b"\xff" if (chunk[2] & 0x80) else b"\x00"), "little", signed=True)
            return value >> 8
        if sample_width == 4:
            return int.from_bytes(chunk, byteorder="little", signed=True) >> 16
        # Unsupported widths fallback to silence.
        return 0

    offset = 0
    for i in range(frame_count):
        mixed = 0
        for _ in range(channels):
            sample = decode_sample(raw[offset : offset + sample_width])
            mixed += sample
            offset += sample_width
        mixed //= channels
        if mixed > 32767:
            mixed = 32767
        elif mixed < -32768:
            mixed = -32768
        out[2 * i : 2 * i + 2] = int(mixed).to_bytes(2, byteorder="little", signed=True)

    return bytes(out)


def _resolve_case_insensitive_file(path: Path) -> Path | None:
    if path.exists() and path.is_file():
        return path
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        return None
    target = path.name.lower()
    for child in parent.iterdir():
        if child.is_file() and child.name.lower() == target:
            return child
    return None


async def play_audio_tool(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    started = time.perf_counter()

    track = arguments.get("track")
    if not isinstance(track, str) or not track.strip():
        return {
            "status": "ignored",
            "reason": "track_missing",
        }

    track = track.strip()
    stream_audio = bool(arguments.get("stream_audio", True))
    requested_format = str(arguments.get("format", "pcm16"))
    requested_sample_rate = int(arguments.get("sample_rate", 16000))
    assets_dir = arguments.get("assets_dir") if isinstance(arguments.get("assets_dir"), str) else None

    resolved_assets_dir = _resolve_assets_dir(assets_dir)
    track_file = _resolve_track_file(track, resolved_assets_dir)
    audio_format = requested_format
    sample_rate = requested_sample_rate
    if track_file and track_file.suffix.lower() in {".wav", ".wave"}:
        try:
            audio_bytes, sample_rate = _read_wav_pcm16(track_file)
            audio_format = "pcm16"
        except Exception:
            _logger.exception(
                "tool_play_audio_wav_decode_failed",
                extra={"track": track, "track_file": str(track_file)},
            )
            audio_bytes = track_file.read_bytes()
    elif track_file:
        audio_bytes = track_file.read_bytes()
    else:
        # Placeholder audio payload for scaffolding when no real asset is present.
        audio_bytes = f"synthetic-audio-for:{track}".encode("utf-8")

    total_bytes = len(audio_bytes)

    await context.emit_event(
        "playback.started",
        {
            "tool": "play_audio",
            "track": track,
            "format": audio_format,
            "sample_rate": sample_rate,
            "total_bytes": total_bytes,
            "streaming": stream_audio,
        },
    )

    chunks_sent = 0
    if stream_audio:
        for index in range(0, total_bytes, _DEFAULT_CHUNK_BYTES):
            chunk = audio_bytes[index : index + _DEFAULT_CHUNK_BYTES]
            chunks_sent += 1
            await context.emit_event(
                "playback.audio.chunk",
                {
                    "track": track,
                    "chunk_index": chunks_sent - 1,
                    "data": base64.b64encode(chunk).decode("ascii"),
                    "format": audio_format,
                    "sample_rate": sample_rate,
                    "is_last": (index + _DEFAULT_CHUNK_BYTES) >= total_bytes,
                },
            )

    await context.emit_event(
        "playback.completed",
        {
            "tool": "play_audio",
            "track": track,
            "streaming": stream_audio,
            "chunks_sent": chunks_sent,
            "total_bytes": total_bytes,
        },
    )

    latency_ms = (time.perf_counter() - started) * 1000
    _logger.info(
        "tool_play_audio_executed",
        extra={
            "session_id": context.session_id,
            "event_type": "tool.play_audio.executed",
            "track": track,
            "streaming": stream_audio,
            "chunks_sent": chunks_sent,
            "total_bytes": total_bytes,
            "latency_ms": round(latency_ms, 3),
        },
    )

    return {
        "status": "ok",
        "tool": "play_audio",
        "track": track,
        "streaming": stream_audio,
        "chunks_sent": chunks_sent,
        "total_bytes": total_bytes,
        "latency_ms": round(latency_ms, 3),
    }
