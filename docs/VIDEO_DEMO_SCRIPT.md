# Video Demo Script (8-12 mins)

## 1) Quick Intro (30s)
- Show repo root.
- State goal: real-time voice assistant with STT + LLM + TTS + tool calling.

## 2) Local Run (1 min)
```bash
source .venv/bin/activate
export OPENAI_API_KEY="sk-..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app
```

In browser:
- Open `http://127.0.0.1:8000/`
- Show Connect / Start Mic controls.

## 3) Basic Voice Turn (1-2 mins)
- Click `Connect`, then `Start Mic`.
- Say: "Hello, what can you do?"
- Show:
  - partial transcript events
  - final transcript
  - assistant text
  - assistant audio completion metrics

## 4) Tool Calling Demo (2 mins)
Prepare:
- Put a WAV track in `assets/` (example: `assets/bbbs.wav`).
- Ensure `PLAY_AUDIO_ASSETS_DIR` points to `assets` (or default `./assets` exists).

Voice query:
- "play track bbbs"

Show in UI logs:
- `tool_call`
- `playback.started`
- `playback.audio.chunk`
- `playback.completed`
- `tool_result`

Mention:
- playback now uses real sample rate (normal speed).

## 5) Observability (1-2 mins)
- Open `http://127.0.0.1:8000/health`
- Open `http://127.0.0.1:8000/metrics`
- Explain:
  - active sessions
  - STT/LLM/TTS latency buckets
  - end-to-end latency
  - cost estimation

## 6) Persistence (1 min)
- Open `data/conversations/<session_id>.jsonl`
- Show saved user + assistant turns.

## 7) Docker Run (1-2 mins)
```bash
docker build -t voice-ai-agent .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." \
  -e PLAY_AUDIO_ASSETS_DIR=/app/assets \
  -v "$(pwd)/assets:/app/assets:ro" \
  -v "$(pwd)/data:/app/data" \
  voice-ai-agent
```

Open UI again and repeat one quick interaction.

## 8) Close with Limitations (30s)
- Mention pending improvements:
  - explicit voice "stop track" interruption
  - broader audio codec support for tool playback
  - deeper automated tests
