"""
Speech-to-Tool-Call — Live demo on Modal.

Records voice from the browser, runs E2E inference on GPU, displays JSON tool call.

Usage:
    modal serve demo.py          # Dev mode (hot reload, temporary URL)
    modal deploy demo.py         # Production (persistent URL)
"""
import modal
import os

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch", "transformers>=4.51.0", "accelerate", "peft", "bitsandbytes",
        "librosa", "soundfile", "numpy", "fastapi", "uvicorn", "python-multipart",
    )
)

model_vol = modal.Volume.from_name("hybrid-model-storage")
app = modal.App("runanywhere-demo")

MODEL_DIR = "/models/best_model_qwen3_finetuned"
DEFAULT_INSTRUCTION = "<|audio|>Convert to tool call JSON:\n"

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Speech-to-Tool-Call</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; min-height: 100vh; padding: 2rem; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; color: #fff; }
  .subtitle { color: #aaa; margin-bottom: 1rem; }
  .stats { display: flex; gap: 1.5rem; margin-bottom: 0.8rem; flex-wrap: wrap; }
  .stat { background: #16213e; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.85rem; }
  .stat b { color: #e94560; }
  .examples { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
  .examples span { color: #aaa; }
  .main { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  @media (max-width: 700px) { .main { grid-template-columns: 1fr; } }
  .panel { background: #16213e; border-radius: 12px; padding: 1.5rem; }
  .record-btn { width: 100%; padding: 1rem; font-size: 1.1rem; border: none; border-radius: 8px;
                cursor: pointer; font-weight: 600; transition: all 0.2s; }
  .record-btn.idle { background: #e94560; color: #fff; }
  .record-btn.idle:hover { background: #c73652; }
  .record-btn.recording { background: #ff6b6b; color: #fff; animation: pulse 1s infinite; }
  .record-btn.processing { background: #555; color: #ccc; cursor: wait; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
  .timer { text-align: center; font-size: 1.5rem; font-variant-numeric: tabular-nums;
           margin: 1rem 0; color: #aaa; }
  .upload-label { display: block; text-align: center; padding: 0.6rem; margin-top: 0.8rem;
                  border: 1px dashed #444; border-radius: 8px; cursor: pointer; color: #888;
                  font-size: 0.85rem; transition: border-color 0.2s; }
  .upload-label:hover { border-color: #e94560; color: #bbb; }
  .upload-label input { display: none; }
  .output-section { margin-bottom: 1rem; }
  .output-section label { font-size: 0.8rem; color: #888; text-transform: uppercase;
                          letter-spacing: 0.05em; display: block; margin-bottom: 0.4rem; }
  .latency { font-size: 0.75rem; color: #e94560; font-weight: 600; float: right; }
  .json-output { background: #0f3460; border-radius: 8px; padding: 1rem; font-family: 'SF Mono',
                 Monaco, 'Cascadia Code', monospace; font-size: 0.85rem; white-space: pre-wrap;
                 word-break: break-word; min-height: 120px; color: #7ec8e3; line-height: 1.5; }
  .text-output { background: #0f3460; border-radius: 8px; padding: 0.8rem; font-size: 0.9rem;
                 color: #ccc; min-height: 2rem; }
  .hidden { display: none; }
  .footer { margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #333;
            font-size: 0.8rem; color: #666; }
  .functions { color: #555; margin-top: 0.5rem; line-height: 1.6; }
</style>
</head>
<body>
<div class="container">
  <h1>Speech-to-Tool-Call</h1>
  <p class="subtitle">Speak a command and get a structured JSON function call — end-to-end from audio, no intermediate text.</p>
  <div class="stats">
    <div class="stat"><b>Architecture</b>&ensp;Whisper-small + Qwen3-0.6B</div>
    <div class="stat"><b>Size</b>&ensp;0.6B params, 730 MB</div>
    <div class="stat"><b>Accuracy</b>&ensp;77% EM on STOP</div>
  </div>
  <p class="examples">Try: <span>"Set an alarm for 7 AM"</span> · <span>"What's the weather?"</span> ·
     <span>"Play some jazz music"</span> · <span>"Navigate to the airport"</span> ·
     <span>"Send a message to John saying I'll be late"</span></p>

  <div class="main">
    <div class="panel">
      <button id="recordBtn" class="record-btn idle" onclick="toggleRecord()">
        Press to Record
      </button>
      <div id="timer" class="timer hidden">0.0s</div>
      <label class="upload-label">
        Or upload an audio file
        <input type="file" id="fileInput" accept="audio/*" onchange="handleFile(event)">
      </label>
    </div>
    <div class="panel">
      <div class="output-section">
        <label>Tool Call Output <span id="e2eLatency" class="latency"></span></label>
        <div id="jsonOutput" class="json-output">Record or upload audio to see results.</div>
      </div>
      <div class="output-section">
        <label>Summary</label>
        <div id="summary" class="text-output">&mdash;</div>
      </div>
      <div class="output-section">
        <label>Whisper ASR Transcript (for reference) <span id="asrLatency" class="latency"></span></label>
        <div id="transcript" class="text-output">&mdash;</div>
      </div>
    </div>
  </div>

  <div class="footer">
    <b>Supported functions:</b>
    <div class="functions">alarm_set, alarm_cancel, alarm_check, alarm_silence, alarm_snooze, alarm_update,
    weather_get_current, navigation_start, navigation_traffic, nav_duration, nav_eta, nav_distance,
    message_send, message_read, message_react, media_play_music, media_playback_control, music_playlist,
    calendar_get_events, timer_set, timer_check, timer_modify, timer_pause, timer_resume, timer_cancel,
    reminder_set, reminder_check, reminder_cancel, reminder_update</div>
  </div>
</div>

<script>
let mediaRecorder, audioChunks = [], recording = false, timerInterval, startTime;
const btn = document.getElementById('recordBtn');
const timerEl = document.getElementById('timer');

async function toggleRecord() {
  if (recording) { stopRecording(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      sendAudio(blob);
    };
    mediaRecorder.start();
    recording = true;
    btn.textContent = 'Stop Recording';
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
  btn.textContent = 'Processing...';
  btn.className = 'record-btn processing';
  timerEl.classList.add('hidden');
}

function handleFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  btn.textContent = 'Processing...';
  btn.className = 'record-btn processing';
  sendAudio(file);
}

async function sendAudio(blob) {
  document.getElementById('jsonOutput').textContent = 'Running E2E inference...';
  document.getElementById('transcript').textContent = 'Waiting for ASR...';
  document.getElementById('summary').textContent = '...';
  document.getElementById('e2eLatency').textContent = '';
  document.getElementById('asrLatency').textContent = '';

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

      // Parse SSE events from buffer
      const lines = buffer.split('\\n');
      buffer = lines.pop(); // keep incomplete line
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'e2e') {
            document.getElementById('jsonOutput').textContent = JSON.stringify(data.tool_call, null, 2);
            document.getElementById('summary').textContent = data.summary;
            document.getElementById('e2eLatency').textContent = data.latency_ms + ' ms';
          } else if (data.type === 'asr') {
            document.getElementById('transcript').textContent = data.transcript;
            document.getElementById('asrLatency').textContent = data.latency_ms + ' ms (total)';
          }
        }
      }
    }
  } catch (err) {
    document.getElementById('jsonOutput').textContent = 'Error: ' + err.message;
  }
  btn.textContent = 'Press to Record';
  btn.className = 'record-btn idle';
}
</script>
</body>
</html>"""


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/models": model_vol},
    timeout=600,
    scaledown_window=300,
)
class Demo:
    @modal.enter()
    def load_model(self):
        """Load models during container startup."""
        import torch
        import torch.nn as nn
        import json
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            WhisperModel, WhisperForConditionalGeneration, WhisperProcessor,
        )
        from peft import LoraConfig, get_peft_model

        class HybridQwen(nn.Module):
            def __init__(self, base_model, whisper_encoder, tokenizer, instruction_prompt,
                         function_names=None):
                super().__init__()
                self.whisper = whisper_encoder
                for param in self.whisper.parameters():
                    param.requires_grad = False

                self.target_dtype = base_model.get_input_embeddings().weight.dtype
                hidden_size = base_model.config.hidden_size
                whisper_dim = whisper_encoder.config.d_model

                self.audio_conv1 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
                self.audio_conv2 = nn.Conv1d(whisper_dim, whisper_dim, kernel_size=3, stride=2, padding=1).to("cuda")
                self.conv_norm = nn.LayerNorm(whisper_dim).to("cuda")
                self.audio_proj = nn.Sequential(
                    nn.Linear(whisper_dim, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                ).to("cuda")

                self.function_names = function_names or []
                num_funcs = len(self.function_names) if self.function_names else 1
                self.function_classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, num_funcs),
                ).to("cuda")

                self.llm = base_model
                self.tokenizer = tokenizer
                self.instruction = instruction_prompt
                self.instruction_ids = tokenizer.encode(
                    self.instruction, add_special_tokens=False, return_tensors="pt"
                ).to("cuda")

            def load_pretrained(self, checkpoint_dir):
                projector_path = os.path.join(checkpoint_dir, "projector.pt")
                if os.path.exists(projector_path):
                    state_dict = torch.load(projector_path, map_location="cuda")
                    self.audio_conv1.load_state_dict(state_dict["audio_conv1"])
                    self.audio_conv2.load_state_dict(state_dict["audio_conv2"])
                    self.conv_norm.load_state_dict(state_dict["conv_norm"])
                    self.audio_proj.load_state_dict(state_dict["audio_proj"])
                    if "function_classifier" in state_dict and self.function_names:
                        self.function_classifier.load_state_dict(state_dict["function_classifier"])
                    if "whisper_top_layers" in state_dict:
                        for idx_str, layer_state in state_dict["whisper_top_layers"].items():
                            self.whisper.layers[int(idx_str)].load_state_dict(layer_state)
                    print("Projector + whisper layers loaded.")

            @torch.no_grad()
            def generate(self, audio_values, max_new_tokens=128):
                enc_out = self.whisper(audio_values).last_hidden_state
                x = enc_out.transpose(1, 2).float()
                x = torch.relu(self.audio_conv1(x))
                x = torch.relu(self.audio_conv2(x))
                x = x.transpose(1, 2)
                x = self.conv_norm(x)
                projected = self.audio_proj(x).to(self.target_dtype)

                bs = projected.shape[0]
                embed_tokens = self.llm.get_input_embeddings()
                instr_ids = self.instruction_ids.expand(bs, -1)
                instr_embeds = embed_tokens(instr_ids)
                current_embeds = torch.cat([projected, instr_embeds], dim=1)

                generated_ids = []
                for _ in range(max_new_tokens):
                    outputs = self.llm(inputs_embeds=current_embeds)
                    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
                    token_id = next_token_id.item()
                    generated_ids.append(token_id)

                    if token_id == self.tokenizer.eos_token_id:
                        break
                    full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    if full_text.strip().endswith('}'):
                        if full_text.count('{') > 0 and full_text.count('{') == full_text.count('}'):
                            break
                    next_token_embed = embed_tokens(next_token_id.unsqueeze(0))
                    current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)

                return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        print("Loading model...")
        config_path = os.path.join(MODEL_DIR, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        instruction_prompt = config.get("instruction_prompt", DEFAULT_INSTRUCTION)
        whisper_model_name = config.get("whisper_model", "openai/whisper-small")
        function_names = config.get("function_names", [])
        base_llm = config.get("base_llm", "unsloth/Qwen3-0.6B")
        lora_r = config.get("lora_r", 32)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
        llm = AutoModelForCausalLM.from_pretrained(base_llm, quantization_config=bnb_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(base_llm)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0, bias="none",
        )
        llm = get_peft_model(llm, lora_config)

        adapter_path = os.path.join(MODEL_DIR, "adapter_model.bin")
        if os.path.exists(adapter_path):
            adapter_state = torch.load(adapter_path, map_location="cpu")
            remapped = {}
            for k, v in adapter_state.items():
                nk = k.replace("base_model.model.base_model.model.model.", "base_model.model.model.")
                nk = nk.replace(".lora_A.weight", ".lora_A.default.weight")
                nk = nk.replace(".lora_B.weight", ".lora_B.default.weight")
                remapped[nk] = v
            llm.load_state_dict(remapped, strict=False)
            print(f"LoRA loaded ({len(remapped)} tensors)")

        self.processor = WhisperProcessor.from_pretrained(whisper_model_name)
        whisper_encoder = WhisperModel.from_pretrained(whisper_model_name).encoder.to("cuda")

        self.hybrid_model = HybridQwen(llm, whisper_encoder, tokenizer, instruction_prompt,
                                       function_names=function_names)
        self.hybrid_model.load_pretrained(MODEL_DIR)
        self.hybrid_model.eval()

        self.whisper_asr = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to("cuda")
        self.whisper_asr.eval()
        print(f"Model ready! {len(function_names)} functions.")

    @modal.asgi_app()
    def serve(self):
        """FastAPI app with custom HTML UI — no Gradio needed."""
        import time
        import torch
        import json
        import tempfile
        import librosa
        import numpy as np
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import HTMLResponse, StreamingResponse

        fast_api = FastAPI()
        hybrid_model = self.hybrid_model
        whisper_asr = self.whisper_asr
        processor = self.processor

        @fast_api.get("/", response_class=HTMLResponse)
        async def index():
            return HTML_PAGE

        @fast_api.post("/predict")
        async def predict(audio: UploadFile = File(...)):
            audio_bytes = await audio.read()

            # Decode audio (supports webm from browser, wav, mp3, etc.)
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                audio_float, sr = librosa.load(tmp.name, sr=16000)

            inputs = processor(audio_float, sampling_rate=16000, return_tensors="pt")
            audio_values = inputs.input_features.to("cuda")

            def generate_events():
                t0 = time.time()

                # E2E prediction — the main result
                prediction = hybrid_model.generate(audio_values)
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

                # Send E2E result immediately
                event = json.dumps({
                    "type": "e2e",
                    "tool_call": tool_call,
                    "summary": summary,
                    "latency_ms": e2e_ms,
                })
                yield f"data: {event}\n\n"

                # Now run ASR (slower, just for reference)
                with torch.no_grad():
                    asr_ids = whisper_asr.generate(audio_values, max_length=448)
                transcript = processor.batch_decode(asr_ids, skip_special_tokens=True)[0]
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

        return fast_api
