from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Voice AI Agent</title>
    <style>
      :root {
        --bg: #f4efe6;
        --card: #fffaf0;
        --ink: #172121;
        --accent: #ce6a85;
        --accent-2: #5b7c8d;
        --muted: #5a6467;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--ink);
        background: radial-gradient(circle at top left, #efe0cc 0%, var(--bg) 45%, #e7f2f5 100%);
        font-family: "Avenir Next", "Segoe UI", sans-serif;
        min-height: 100vh;
        display: grid;
        place-items: center;
        padding: 20px;
      }
      .panel {
        width: min(920px, 100%);
        background: var(--card);
        border: 2px solid rgba(23, 33, 33, 0.14);
        border-radius: 20px;
        box-shadow: 0 22px 60px rgba(23, 33, 33, 0.12);
        overflow: hidden;
      }
      header {
        padding: 18px 20px;
        background: linear-gradient(110deg, var(--accent), var(--accent-2));
        color: #fff;
      }
      h1 {
        margin: 0;
        font-size: 20px;
      }
      .sub {
        opacity: 0.92;
        margin-top: 4px;
        font-size: 13px;
      }
      .controls {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 14px 16px;
        border-bottom: 1px solid rgba(23, 33, 33, 0.1);
      }
      button {
        border: 0;
        border-radius: 999px;
        padding: 10px 14px;
        font-weight: 700;
        cursor: pointer;
        color: #fff;
        background: var(--accent-2);
      }
      button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }
      #btn-mic {
        background: var(--accent);
      }
      .status {
        margin-left: auto;
        align-self: center;
        font-size: 13px;
        color: var(--muted);
      }
      .log {
        height: 56vh;
        overflow: auto;
        padding: 12px 16px 16px;
        font-size: 14px;
      }
      .event {
        padding: 9px 10px;
        border-radius: 10px;
        margin: 6px 0;
        background: rgba(91, 124, 141, 0.08);
        border: 1px solid rgba(91, 124, 141, 0.18);
      }
      .event .type {
        font-weight: 800;
        font-size: 12px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        color: #32444d;
      }
      .event pre {
        margin: 5px 0 0;
        white-space: pre-wrap;
        word-break: break-word;
        font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
      }
    </style>
  </head>
  <body>
    <div class="panel">
      <header>
        <h1>Voice AI Agent</h1>
        <div class="sub">Browser mic -> WebSocket -> FastAPI session pipeline</div>
      </header>
      <div class="controls">
        <button id="btn-connect">Connect</button>
        <button id="btn-disconnect" disabled>Disconnect</button>
        <button id="btn-mic" disabled>Start Mic</button>
        <button id="btn-stop" disabled>Stop Mic</button>
        <button id="btn-clear">Clear Log</button>
        <div class="status" id="status">Disconnected</div>
      </div>
      <div class="log" id="log"></div>
    </div>

    <script>
      const statusEl = document.getElementById("status");
      const logEl = document.getElementById("log");
      const btnConnect = document.getElementById("btn-connect");
      const btnDisconnect = document.getElementById("btn-disconnect");
      const btnMic = document.getElementById("btn-mic");
      const btnStop = document.getElementById("btn-stop");
      const btnClear = document.getElementById("btn-clear");

      let ws = null;
      let audioCtx = null;
      let mediaStream = null;
      let sourceNode = null;
      let processorNode = null;
      let assistantAudioBuffer = [];
      let assistantAudioSampleRate = 16000;
      let playbackAudioBuffer = [];
      let playbackAudioSampleRate = 16000;
      let isMicStreaming = false;
      let requestedDisconnect = false;
      let recognition = null;
      let browserSttActive = false;

      function logEvent(type, payload) {
        const row = document.createElement("div");
        row.className = "event";
        row.innerHTML = `<div class="type">${type}</div><pre>${JSON.stringify(payload, null, 2)}</pre>`;
        logEl.appendChild(row);
        logEl.scrollTop = logEl.scrollHeight;
      }

      function setStatus(text) {
        statusEl.textContent = text;
      }

      function updateButtons() {
        const connected = !!ws && ws.readyState === WebSocket.OPEN;
        btnConnect.disabled = connected;
        btnDisconnect.disabled = !connected;
        btnMic.disabled = !connected || isMicStreaming;
        btnStop.disabled = !isMicStreaming;
      }

      function supportsBrowserStt() {
        return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
      }

      function toBase64(uint8) {
        let binary = "";
        const chunk = 0x8000;
        for (let i = 0; i < uint8.length; i += chunk) {
          const sub = uint8.subarray(i, i + chunk);
          binary += String.fromCharCode(...sub);
        }
        return btoa(binary);
      }

      function fromBase64(b64) {
        const bin = atob(b64);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out;
      }

      function floatToPcm16(float32) {
        const out = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return new Uint8Array(out.buffer);
      }

      function downsampleTo16k(input, srcRate) {
        const targetRate = 16000;
        if (srcRate === targetRate) return input;
        const ratio = srcRate / targetRate;
        const outLen = Math.round(input.length / ratio);
        const out = new Float32Array(outLen);
        let offsetResult = 0;
        let offsetBuffer = 0;
        while (offsetResult < out.length) {
          const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
          let accum = 0;
          let count = 0;
          for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i++) {
            accum += input[i];
            count++;
          }
          out[offsetResult] = count > 0 ? accum / count : 0;
          offsetResult++;
          offsetBuffer = nextOffsetBuffer;
        }
        return out;
      }

      async function playAssistantPcm16(pcmBytes, sampleRate = 16000) {
        if (!pcmBytes || pcmBytes.length < 2) return;
        if (!audioCtx) {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        await audioCtx.resume();
        const samples = new Int16Array(pcmBytes.buffer, pcmBytes.byteOffset, Math.floor(pcmBytes.byteLength / 2));
        const audioBuffer = audioCtx.createBuffer(1, samples.length, sampleRate);
        const channel = audioBuffer.getChannelData(0);
        for (let i = 0; i < samples.length; i++) {
          channel[i] = samples[i] / 32768;
        }
        const src = audioCtx.createBufferSource();
        src.buffer = audioBuffer;
        src.connect(audioCtx.destination);
        src.start();
      }

      function connectWs() {
        const protocol = window.location.protocol === "https:" ? "wss" : "ws";
        const url = `${protocol}://${window.location.host}/ws/talk`;
        requestedDisconnect = false;
        ws = new WebSocket(url);
        ws.onopen = () => {
          setStatus("Connected");
          logEvent("ws.connected", { url });
          updateButtons();
        };
        ws.onerror = (event) => {
          logEvent("ws.error", { detail: String(event) });
        };
        ws.onclose = (event) => {
          setStatus(`Disconnected (${event.code})`);
          logEvent("ws.closed", {
            code: event.code,
            reason: event.reason || null,
            initiator: requestedDisconnect ? "client" : "server_or_network",
          });
          ws = null;
          stopMic({ suppressLog: true }).catch(() => {});
          requestedDisconnect = false;
          updateButtons();
        };
        ws.onmessage = async (event) => {
          let message;
          try {
            message = JSON.parse(event.data);
          } catch {
            return;
          }
          const t = message.type || "unknown";
          const payload = (message && typeof message.payload === "object" && message.payload) || {};
          if (t === "assistant_audio.chunk" && typeof payload.data === "string") {
            if (Number.isFinite(payload.sample_rate) && payload.sample_rate > 0) {
              assistantAudioSampleRate = payload.sample_rate;
            }
            assistantAudioBuffer.push(fromBase64(payload.data));
            return;
          }
          if (t === "playback.audio.chunk" && typeof payload.data === "string") {
            if (Number.isFinite(payload.sample_rate) && payload.sample_rate > 0) {
              playbackAudioSampleRate = payload.sample_rate;
            }
            playbackAudioBuffer.push(fromBase64(payload.data));
            return;
          }
          if (t === "assistant_audio.completed") {
            const totalLen = assistantAudioBuffer.reduce((n, p) => n + p.length, 0);
            const merged = new Uint8Array(totalLen);
            let off = 0;
            for (const part of assistantAudioBuffer) {
              merged.set(part, off);
              off += part.length;
            }
            assistantAudioBuffer = [];
            await playAssistantPcm16(merged, assistantAudioSampleRate);
            logEvent(t, payload);
            return;
          }
          if (t === "playback.completed") {
            const totalLen = playbackAudioBuffer.reduce((n, p) => n + p.length, 0);
            if (totalLen > 0) {
              const merged = new Uint8Array(totalLen);
              let off = 0;
              for (const part of playbackAudioBuffer) {
                merged.set(part, off);
                off += part.length;
              }
              playbackAudioBuffer = [];
              await playAssistantPcm16(merged, playbackAudioSampleRate);
            }
            logEvent(t, payload);
            return;
          }
          if (
            t === "transcript.partial" ||
            t === "transcript.final" ||
            t === "assistant_text" ||
            t === "error" ||
            t === "tool_call" ||
            t === "tool_result" ||
            t === "playback.started" ||
            t === "playback.completed"
          ) {
            logEvent(t, payload);
          }
        };
      }

      async function startMic() {
        if (!ws || ws.readyState !== WebSocket.OPEN || isMicStreaming) return;
        if (!audioCtx) {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        await audioCtx.resume();
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        sourceNode = audioCtx.createMediaStreamSource(mediaStream);
        processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
        processorNode.onaudioprocess = (event) => {
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          const input = event.inputBuffer.getChannelData(0);
          const downsampled = downsampleTo16k(input, audioCtx.sampleRate);
          const pcm16 = floatToPcm16(downsampled);
          const payload = {
            type: "audio_chunk",
            format: "pcm16",
            sample_rate: 16000,
            data: toBase64(pcm16),
          };
          ws.send(JSON.stringify(payload));
        };
        sourceNode.connect(processorNode);
        processorNode.connect(audioCtx.destination);
        isMicStreaming = true;
        setStatus("Connected • Mic streaming");
        logEvent("mic.started", { sample_rate: audioCtx.sampleRate });
        startBrowserStt();
        updateButtons();
      }

      async function stopMic({ suppressLog = false } = {}) {
        stopBrowserStt();
        const wasStreaming = isMicStreaming;
        if (processorNode) {
          processorNode.disconnect();
          processorNode.onaudioprocess = null;
          processorNode = null;
        }
        if (sourceNode) {
          sourceNode.disconnect();
          sourceNode = null;
        }
        if (mediaStream) {
          mediaStream.getTracks().forEach((t) => t.stop());
          mediaStream = null;
        }
        isMicStreaming = false;
        if (ws && ws.readyState === WebSocket.OPEN) {
          setStatus("Connected");
        }
        if (wasStreaming && !suppressLog) {
          logEvent("mic.stopped", {});
        }
        updateButtons();
      }

      function startBrowserStt() {
        if (!supportsBrowserStt()) {
          logEvent("stt.browser.unavailable", { detail: "SpeechRecognition not supported in this browser." });
          return;
        }
        if (browserSttActive) return;
        const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new Ctor();
        recognition.lang = "en-US";
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.onresult = (event) => {
          let finalText = "";
          let interimText = "";
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0]?.transcript || "";
            if (event.results[i].isFinal) {
              finalText += transcript;
            } else {
              interimText += transcript;
            }
          }
          if (interimText.trim()) {
            logEvent("transcript.partial.browser", { text: interimText.trim() });
          }
          if (finalText.trim()) {
            const text = finalText.trim();
            logEvent("transcript.final.browser", { text });
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send(
                JSON.stringify({
                  type: "control",
                  action: "text",
                  payload: { text },
                })
              );
            }
          }
        };
        recognition.onerror = (event) => {
          logEvent("stt.browser.error", { error: event.error || "unknown" });
        };
        recognition.onend = () => {
          if (browserSttActive) {
            try {
              recognition.start();
            } catch {
              // no-op
            }
          }
        };
        try {
          recognition.start();
          browserSttActive = true;
          logEvent("stt.browser.started", {});
        } catch (err) {
          logEvent("stt.browser.error", { error: String(err) });
        }
      }

      function stopBrowserStt() {
        browserSttActive = false;
        if (recognition) {
          try {
            recognition.onend = null;
            recognition.stop();
          } catch {
            // no-op
          }
          recognition = null;
          logEvent("stt.browser.stopped", {});
        }
      }

      btnConnect.onclick = () => connectWs();
      btnDisconnect.onclick = async () => {
        await stopMic();
        requestedDisconnect = true;
        if (ws) ws.close(1000, "client_disconnect");
      };
      btnMic.onclick = async () => {
        try {
          await startMic();
        } catch (err) {
          logEvent("mic.error", { error: String(err) });
          setStatus("Mic error");
        }
      };
      btnStop.onclick = () => stopMic();
      btnClear.onclick = () => (logEl.innerHTML = "");

      updateButtons();
    </script>
  </body>
</html>"""
