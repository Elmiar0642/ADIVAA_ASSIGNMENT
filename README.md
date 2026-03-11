# Voice AI Agent (FastAPI)

Real-time voice AI agent using FastAPI + WebSockets with:
- async session pipeline
- OpenAI Realtime STT
- GPT reasoning + tool calling
- streaming TTS
- structured JSON logs
- metrics and conversation persistence

## Assignment Coverage

Status:
- Completed: core API, session pipeline, concurrency, logging, metrics, STT/LLM/TTS integration, tool registry, Docker, mic client.
- Partially complete: advanced interruption controls (`stop track`) and extensive test suite polish.

## Project Structure

```text
app/
  main.py
  api/
    ws.py
    health.py
    metrics.py
    root.py
  core/
    config.py
    logging.py
    metrics.py
    session.py
    session_manager.py
    conversation_store.py
  services/
    orchestrator.py
  openai/
    realtime_client.py
    stt_client.py
    llm_client.py
    tts_client.py
  tools/
    registry.py
    play_audio.py
  models/
    events.py
ws_mic_client.py
Dockerfile
```

## Requirements

- Python `3.11+` (project currently allows up to `<3.15`)
- OpenAI API key

## Environment Variables

Required:
- `OPENAI_API_KEY`

Common optional:
- `OPENAI_REALTIME_MODEL` (default: `gpt-realtime`)
- `OPENAI_TRANSCRIPTION_MODEL` (default: `gpt-4o-mini-transcribe`)
- `OPENAI_LLM_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_TTS_MODEL` (default: `gpt-4o-mini-tts`)
- `OPENAI_TTS_VOICE` (default: `nova`)
- `OPENAI_TTS_SAMPLE_RATE` (default: `24000`)
- `PLAY_AUDIO_ASSETS_DIR` (default fallback: `./assets`)
- `CONVERSATION_LOG_DIR` (default: `data/conversations`)

## Run Locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY="sk-..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
```

Open:
- `http://127.0.0.1:8000/` (Web UI)
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/metrics`

## Run with Docker

Build:
```bash
docker build -t voice-ai-agent .
```

Run:
```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." \
  -e PLAY_AUDIO_ASSETS_DIR=/app/assets \
  -v "$(pwd)/assets:/app/assets:ro" \
  -v "$(pwd)/data:/app/data" \
  voice-ai-agent
```

Notes:
- `assets` is mounted read-only for track playback.
- `data` is mounted so conversation JSONL logs persist outside container.

## WebSocket Contract (`/ws/talk`)

Audio chunk:
```json
{
  "type": "audio_chunk",
  "format": "pcm16",
  "sample_rate": 16000,
  "data": "<base64>"
}
```

Control message:
```json
{
  "type": "control",
  "action": "text",
  "payload": { "text": "hello" }
}
```

## Endpoints

- `GET /` browser demo UI
- `GET /health` service health
- `GET /metrics` metrics snapshot (JSON)
- `WS /ws/talk` bidirectional streaming

## Data Persistence

Conversation turns are stored as JSONL under `data/conversations/<session_id>.jsonl`.

Shape:
```json
{
  "timestamp": "...",
  "session_id": "...",
  "role": "user|assistant",
  "text": "...",
  "metadata": {}
}
```

## Architecture Walkthrough

Use this file to explain code during review/interview:
- [Code Walkthrough](docs/CODE_WALKTHROUGH.md)

## CLI Simulator (Optional)

```bash
python3 ws_mic_client.py --url ws://localhost:8000/ws/talk
```

Optional local playback dependency:
```bash
pip install simpleaudio
```
