# Code Walkthrough

## 1) Entry and Routing
- `app/main.py`
  - Creates FastAPI app.
  - Configures structured logging.
  - Mounts routes for `/`, `/health`, `/metrics`, `/ws/talk`.

- `app/api/ws.py`
  - Accepts WebSocket connections.
  - Parses incoming events (`audio_chunk`, `control`).
  - Creates one `Session` per connection via `SessionManager`.
  - Routes audio/control to orchestrator.

## 2) Session Lifecycle and Concurrency
- `app/core/session_manager.py`
  - Tracks active sessions.
  - Adds/removes sessions safely with async lock.
  - Updates active session gauge.

- `app/core/session.py`
  - Core real-time pipeline.
  - Queues:
    - `audio_input_queue`
    - `transcript_queue`
    - `llm_queue`
    - `tts_queue`
    - internal tool queue
  - Workers:
    - audio ingestion
    - transcription handling
    - LLM reasoning
    - tool execution
    - TTS streaming
  - Handles teardown and task cancellation on disconnect.

## 3) OpenAI Integration
- `app/openai/realtime_client.py`
  - Async WebSocket client for OpenAI Realtime.
  - Sends input audio frames.
  - Receives transcript and model events via callbacks.
  - Handles reconnect/close paths.

- `app/openai/llm_client.py`
  - GPT chat completion for reasoning.
  - Supports tool calling (`play_audio`).
  - Includes local fallback behavior when API fails.

- `app/openai/tts_client.py`
  - Streams TTS audio chunks.
  - Uses `gpt-4o-mini-tts`.
  - Default voice set to `nova`.

## 4) Tools
- `app/tools/registry.py`
  - Registers and executes tools by name.

- `app/tools/play_audio.py`
  - Resolves requested track from assets directory.
  - Emits playback start/chunk/completed events.
  - WAV conversion path preserves real sample rate to avoid slow playback.

## 5) Observability
- `app/core/logging.py`
  - JSON log formatter with structured fields.

- `app/core/metrics.py`
  - Tracks counters/gauges and latency distributions.
  - Exposes per-stage and cost summaries.

- `app/api/metrics.py`
  - Returns metrics JSON.

## 6) Persistence
- `app/core/conversation_store.py`
  - Appends user and assistant turns as JSONL.
  - One file per session.

## 7) Browser Demo UI
- `app/api/root.py`
  - Serves single-page UI.
  - Captures browser mic audio and sends to `/ws/talk`.
  - Plays assistant TTS and tool playback audio streams.
  - Logs transcript/tool/playback events live.
