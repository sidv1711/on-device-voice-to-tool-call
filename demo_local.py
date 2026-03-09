"""
Local FastAPI demo — runs on Apple Silicon via MLX. No Modal, no CUDA.

Same UI as demo.py but using MLX inference backend.

Usage:
    python demo_local.py                              # Default model paths
    python demo_local.py --model-dir ./model_export   # Custom path
    python demo_local.py --quantized                  # Use 4-bit MLX model

Then open http://localhost:8000 in your browser.
"""
import argparse
import json
import os
import tempfile
import time

import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from mlx_model import HybridQwenMLX

# ---------------------------------------------------------------------------
# HTML UI (same as Modal demo.py)
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RunAnywhere — Speech to Tool Call</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f7; color: #1d1d1f; min-height: 100vh; }

  /* Header bar */
  .topbar { background: #fff; border-bottom: 1px solid #e5e5e5; padding: 0.8rem 2rem;
            display: flex; align-items: center; gap: 0.75rem; }
  .topbar-icon { width: 36px; height: 36px; background: #1d1d1f; border-radius: 10px;
                 display: flex; align-items: center; justify-content: center; }
  .topbar-icon svg { width: 20px; height: 20px; fill: #fff; }
  .topbar-text h2 { font-size: 0.95rem; font-weight: 600; }
  .topbar-text p { font-size: 0.75rem; color: #86868b; }

  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 2rem 3rem; }

  h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.4rem; }
  .subtitle { color: #6e6e73; font-size: 0.95rem; margin-bottom: 1.5rem; line-height: 1.5; }

  .pills { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .pill { background: #fff; border: 1px solid #e5e5e5; padding: 0.35rem 0.75rem;
          border-radius: 100px; font-size: 0.78rem; color: #6e6e73; }
  .pill b { color: #1d1d1f; font-weight: 600; }
  .pill.green { background: #e8f5e9; border-color: #c8e6c9; color: #2e7d32; }

  /* Card grid */
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }

  .card { background: #fff; border: 1px solid #e5e5e5; border-radius: 14px; padding: 1.5rem;
          transition: box-shadow 0.2s; }
  .card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
  .card-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; }
  .card-icon { width: 28px; height: 28px; background: #f5f5f7; border-radius: 8px;
               display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .card-icon svg { width: 16px; height: 16px; fill: #6e6e73; }
  .card-title { font-size: 0.9rem; font-weight: 600; }
  .card-badge { margin-left: auto; font-size: 0.7rem; font-weight: 600; color: #e94560;
                background: #fef0f2; padding: 0.2rem 0.5rem; border-radius: 6px; }

  /* Audio input card */
  .waveform { background: #f5f5f7; border-radius: 10px; height: 80px; margin-bottom: 1rem;
              display: flex; align-items: center; justify-content: center; overflow: hidden;
              position: relative; }
  .waveform-placeholder { color: #c7c7cc; font-size: 0.8rem; }
  .waveform canvas { width: 100%; height: 100%; }
  .timer-display { position: absolute; right: 12px; top: 50%; transform: translateY(-50%);
                   font-size: 1.1rem; font-weight: 600; color: #1d1d1f;
                   font-variant-numeric: tabular-nums; }
  .btn-row { display: flex; gap: 0.75rem; }
  .record-btn { flex: 1; padding: 0.7rem 1rem; font-size: 0.88rem; border: none; border-radius: 10px;
                cursor: pointer; font-weight: 600; transition: all 0.15s;
                display: flex; align-items: center; justify-content: center; gap: 0.4rem; }
  .record-btn svg { width: 16px; height: 16px; fill: currentColor; }
  .record-btn.idle { background: #1d1d1f; color: #fff; }
  .record-btn.idle:hover { background: #333; }
  .record-btn.recording { background: #e94560; color: #fff; animation: pulse 1.2s infinite; }
  .record-btn.processing { background: #d1d1d6; color: #8e8e93; cursor: wait; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
  .upload-btn { padding: 0.7rem 1.2rem; font-size: 0.88rem; border: 1px solid #e5e5e5;
                border-radius: 10px; cursor: pointer; font-weight: 500; background: #fff;
                color: #1d1d1f; transition: all 0.15s;
                display: flex; align-items: center; gap: 0.4rem; }
  .upload-btn:hover { background: #f5f5f7; border-color: #d1d1d6; }
  .upload-btn svg { width: 16px; height: 16px; fill: #6e6e73; }
  .upload-btn input { display: none; }

  /* Output areas */
  .output-area { background: #f5f5f7; border: 1px dashed #d1d1d6; border-radius: 10px;
                 padding: 1rem; min-height: 140px; display: flex; align-items: center;
                 justify-content: center; transition: all 0.2s; }
  .output-area.has-content { border-style: solid; border-color: #e5e5e5; align-items: flex-start;
                             justify-content: flex-start; }
  .output-placeholder { color: #c7c7cc; font-size: 0.85rem; }
  .json-output { font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Fira Code', monospace;
                 font-size: 0.82rem; white-space: pre-wrap; word-break: break-word;
                 color: #1d1d1f; line-height: 1.6; width: 100%; }
  .text-output { font-size: 0.9rem; color: #1d1d1f; line-height: 1.5; width: 100%; }

  /* Summary card (spans full width) */
  .card.full { grid-column: 1 / -1; }
  .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 720px) { .summary-grid { grid-template-columns: 1fr; } }
  .summary-item label { display: block; font-size: 0.72rem; color: #86868b;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem;
                        font-weight: 500; }
  .summary-item .value { font-size: 0.9rem; color: #1d1d1f; min-height: 1.4em; }

  /* Examples */
  .examples { margin-bottom: 1.5rem; }
  .examples-label { font-size: 0.78rem; color: #86868b; text-transform: uppercase;
                    letter-spacing: 0.05em; font-weight: 500; margin-bottom: 0.5rem; }
  .example-chips { display: flex; gap: 0.4rem; flex-wrap: wrap; }
  .chip { background: #fff; border: 1px solid #e5e5e5; padding: 0.3rem 0.7rem;
          border-radius: 100px; font-size: 0.8rem; color: #6e6e73; cursor: default;
          transition: border-color 0.15s; }
  .chip:hover { border-color: #1d1d1f; color: #1d1d1f; }

  .hidden { display: none; }
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-icon">
    <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
  </div>
  <div class="topbar-text">
    <h2>RunAnywhere</h2>
    <p>Speech-to-Tool-Call</p>
  </div>
</div>

<div class="container">
  <h1>Voice Command to JSON Tool Call</h1>
  <p class="subtitle">Speak a command and get a structured JSON function call — end-to-end from audio, no intermediate text. Running locally on Apple Silicon via MLX.</p>

  <div class="pills">
    <div class="pill"><b>Model</b>&ensp;Whisper-small + Qwen3-0.6B</div>
    <div class="pill"><b>Params</b>&ensp;0.6B</div>
    <div class="pill"><b>Accuracy</b>&ensp;77% EM</div>
    <div class="pill green">On-Device MLX</div>
  </div>

  <div class="examples">
    <div class="examples-label">Try saying</div>
    <div class="example-chips">
      <span class="chip">"Set an alarm for 7 AM"</span>
      <span class="chip">"What's the weather?"</span>
      <span class="chip">"Play some jazz music"</span>
      <span class="chip">"Navigate to the airport"</span>
      <span class="chip">"Send a message to John"</span>
    </div>
  </div>

  <div class="grid">
    <!-- Audio Input -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon">
          <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
        </div>
        <span class="card-title">Audio Input</span>
      </div>
      <div class="waveform" id="waveformArea">
        <canvas id="waveformCanvas"></canvas>
        <span class="waveform-placeholder" id="waveformPlaceholder">Audio waveform will appear here</span>
        <span class="timer-display hidden" id="timer">0.0s</span>
      </div>
      <div class="btn-row">
        <button id="recordBtn" class="record-btn idle" onclick="toggleRecord()">
          <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
          Start Recording
        </button>
        <label class="upload-btn">
          <svg viewBox="0 0 24 24"><path d="M9 16h6v-6h4l-7-7-7 7h4v6zm-4 2h14v2H5v-2z"/></svg>
          Upload
          <input type="file" id="fileInput" accept="audio/*" onchange="handleFile(event)">
        </label>
      </div>
    </div>

    <!-- Tool Call Output -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon">
          <svg viewBox="0 0 24 24"><path d="M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0L19.2 12l-4.6-4.6L16 6l6 6-6 6-1.4-1.4z"/></svg>
        </div>
        <span class="card-title">Tool Call Output</span>
        <span class="card-badge" id="e2eLatency"></span>
      </div>
      <div class="output-area" id="jsonArea">
        <span class="output-placeholder" id="jsonPlaceholder">Tool call JSON will appear after processing</span>
        <div class="json-output hidden" id="jsonOutput"></div>
      </div>
    </div>

    <!-- Transcript -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon">
          <svg viewBox="0 0 24 24"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm-1 7V3.5L18.5 9H13zM6 20V4h5v7h7v9H6z"/></svg>
        </div>
        <span class="card-title">Transcript</span>
        <span class="card-badge" id="asrLatency"></span>
      </div>
      <div class="output-area" id="transcriptArea">
        <span class="output-placeholder" id="transcriptPlaceholder">Transcript will appear after processing</span>
        <div class="text-output hidden" id="transcript"></div>
      </div>
    </div>

    <!-- Summary -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon">
          <svg viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg>
        </div>
        <span class="card-title">Summary</span>
      </div>
      <div class="summary-grid">
        <div class="summary-item">
          <label>Function</label>
          <div class="value" id="summaryFunc">--</div>
        </div>
        <div class="summary-item">
          <label>Arguments</label>
          <div class="value" id="summaryArgs">--</div>
        </div>
      </div>
    </div>
  </div>

  <div style="margin-top:1rem; color:#86868b; font-size:0.72rem; line-height:1.8;">
    <span style="font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">30 supported functions</span>&ensp;
    alarm_set &middot; alarm_cancel &middot; weather_get_current &middot; navigation_start &middot; navigation_traffic &middot;
    message_send &middot; message_read &middot; media_play_music &middot; media_playback_control &middot; media_volume_control &middot;
    media_podcast_play &middot; calendar_create_event &middot; calendar_get_events &middot; email_check &middot; email_send &middot;
    info_search &middot; info_get_news &middot; info_get_time &middot; shopping_add_to_list &middot; shopping_read_list &middot;
    smart_light_control &middot; smart_plug_control &middot; smart_vacuum_control &middot; travel_book_ride &middot;
    utility_calculate &middot; utility_convert &middot; utility_define &middot; food_find_restaurant &middot; food_order &middot; finance_check_stocks
  </div>
</div>

<script>
let mediaRecorder, audioChunks = [], recording = false, timerInterval, startTime;
let audioContext, analyser, animFrame;
const btn = document.getElementById('recordBtn');
const timerEl = document.getElementById('timer');
const canvas = document.getElementById('waveformCanvas');
const canvasCtx = canvas.getContext('2d');

function drawWaveform(analyserNode) {
  const bufLen = analyserNode.frequencyBinCount;
  const data = new Uint8Array(bufLen);
  function draw() {
    animFrame = requestAnimationFrame(draw);
    analyserNode.getByteTimeDomainData(data);
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = canvas.offsetHeight * 2;
    canvasCtx.clearRect(0, 0, w, h);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = '#1d1d1f';
    canvasCtx.beginPath();
    const sliceWidth = w / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
      const v = data[i] / 128.0;
      const y = v * h / 2;
      i === 0 ? canvasCtx.moveTo(x, y) : canvasCtx.lineTo(x, y);
      x += sliceWidth;
    }
    canvasCtx.lineTo(w, h / 2);
    canvasCtx.stroke();
  }
  draw();
}

async function toggleRecord() {
  if (recording) { stopRecording(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    document.getElementById('waveformPlaceholder').classList.add('hidden');
    canvas.classList.remove('hidden');
    drawWaveform(analyser);

    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(animFrame);
      audioContext.close();
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      sendAudio(blob);
    };
    mediaRecorder.start();
    recording = true;
    btn.innerHTML = '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg> Stop Recording';
    btn.className = 'record-btn recording';
    timerEl.classList.remove('hidden');
    startTime = Date.now();
    timerInterval = setInterval(() => {
      timerEl.textContent = ((Date.now() - startTime) / 1000).toFixed(1) + 's';
    }, 100);
  } catch (err) {
    alert('Microphone access denied. Please allow microphone access and try again.');
  }
}

function stopRecording() {
  mediaRecorder.stop();
  recording = false;
  clearInterval(timerInterval);
  btn.innerHTML = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/></svg> Processing...';
  btn.className = 'record-btn processing';
  timerEl.classList.add('hidden');
}

function handleFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  btn.innerHTML = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/></svg> Processing...';
  btn.className = 'record-btn processing';
  sendAudio(file);
}

function showOutput(id, placeholderId, areaId, text) {
  document.getElementById(placeholderId).classList.add('hidden');
  const el = document.getElementById(id);
  el.classList.remove('hidden');
  el.textContent = text;
  document.getElementById(areaId).classList.add('has-content');
}

async function sendAudio(blob) {
  // Reset outputs
  document.getElementById('jsonPlaceholder').textContent = 'Running inference...';
  document.getElementById('jsonPlaceholder').classList.remove('hidden');
  document.getElementById('jsonOutput').classList.add('hidden');
  document.getElementById('jsonArea').classList.remove('has-content');
  document.getElementById('transcriptPlaceholder').textContent = 'Waiting for ASR...';
  document.getElementById('transcriptPlaceholder').classList.remove('hidden');
  document.getElementById('transcript').classList.add('hidden');
  document.getElementById('transcriptArea').classList.remove('has-content');
  document.getElementById('e2eLatency').textContent = '';
  document.getElementById('asrLatency').textContent = '';
  document.getElementById('summaryFunc').textContent = '...';
  document.getElementById('summaryArgs').textContent = '...';

  const formData = new FormData();
  formData.append('audio', blob);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'e2e') {
            showOutput('jsonOutput', 'jsonPlaceholder', 'jsonArea',
                       JSON.stringify(data.tool_call, null, 2));
            document.getElementById('e2eLatency').textContent = data.latency_ms + ' ms';
            // Update summary
            if (data.tool_call && data.tool_call.function) {
              document.getElementById('summaryFunc').textContent = data.tool_call.function;
              const args = data.tool_call.arguments;
              if (args && Object.keys(args).length > 0) {
                document.getElementById('summaryArgs').textContent =
                  Object.entries(args).map(([k,v]) => k + ': ' + v).join(', ');
              } else {
                document.getElementById('summaryArgs').textContent = 'None';
              }
            } else {
              document.getElementById('summaryFunc').textContent = 'Parse error';
              document.getElementById('summaryArgs').textContent = '--';
            }
          } else if (data.type === 'asr') {
            showOutput('transcript', 'transcriptPlaceholder', 'transcriptArea',
                       data.transcript);
            document.getElementById('asrLatency').textContent = data.latency_ms + ' ms';
          }
        }
      }
    }
  } catch (err) {
    document.getElementById('jsonPlaceholder').textContent = 'Error: ' + err.message;
  }
  btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg> Start Recording';
  btn.className = 'record-btn idle';
}
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Speech-to-Tool-Call (Local MLX)")

# Global model reference (loaded at startup)
hybrid_model: HybridQwenMLX = None


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    # Decode audio: convert webm/mp3/etc to wav via ffmpeg, then load
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in.flush()
        wav_path = tmp_in.name + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in.name, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, check=True,
        )
        audio_float, sr = librosa.load(wav_path, sr=16000)
    finally:
        os.unlink(tmp_in.name)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    def generate_events():
        t0 = time.time()

        # E2E prediction via MLX
        prediction = hybrid_model.generate(audio_array=audio_float, sr=sr)
        e2e_ms = int((time.time() - t0) * 1000)

        try:
            tool_call = json.loads(prediction)
        except json.JSONDecodeError:
            tool_call = {"raw_output": prediction, "parse_error": True}

        if isinstance(tool_call, dict) and "function" in tool_call:
            func = tool_call["function"]
            args = tool_call.get("arguments", {})
            summary = f"Function: {func}"
            if args:
                summary += "\nArguments: " + ", ".join(
                    f"{k}={v}" for k, v in args.items()
                )
        else:
            summary = json.dumps(tool_call, indent=2)

        event = json.dumps({
            "type": "e2e",
            "tool_call": tool_call,
            "summary": summary,
            "latency_ms": e2e_ms,
        })
        yield f"data: {event}\n\n"

        # ASR transcription (for reference)
        t1 = time.time()
        transcript = hybrid_model.transcribe(audio_array=audio_float, sr=sr)
        total_ms = int((time.time() - t0) * 1000)

        event = json.dumps({
            "type": "asr",
            "transcript": transcript,
            "latency_ms": total_ms,
        })
        yield f"data: {event}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local MLX demo server")
    parser.add_argument("--model-dir", default="./model_export",
                        help="Directory with model artifacts")
    parser.add_argument("--qwen-mlx", default=None,
                        help="Path to MLX-converted Qwen3 model")
    parser.add_argument("--quantized", action="store_true",
                        help="Use 4-bit quantized MLX model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global hybrid_model
    print("Loading model (this takes ~30s on first run)...")
    hybrid_model = HybridQwenMLX.load(
        args.model_dir,
        qwen_mlx_path=args.qwen_mlx,
        quantized=args.quantized,
    )
    print(f"Model loaded! Starting server on http://localhost:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
