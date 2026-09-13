"""
Modern Interactive Web Playground for Nool-Alpha-100M.
Self-contained Python HTTP server with real-time token streaming (SSE),
checkpoint switching, parameter tuning, and multi-domain presets.
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import parse_qs, urlparse

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nool_alpha.infer import NoolAlphaInference, DEFAULT_MODEL_DIR

# Global shared inference engine
inference_engine: NoolAlphaInference = None
engine_lock = threading.Lock()

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nool-Alpha-100M Playground</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card-bg: rgba(22, 27, 46, 0.85);
            --card-border: rgba(99, 102, 241, 0.2);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent: #ec4899;
            --accent-cyan: #06b6d4;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --code-bg: #080c16;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.12) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            border-bottom: 1px solid var(--card-border);
            backdrop-filter: blur(12px);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(11, 15, 25, 0.8);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .brand-logo {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.2rem;
            color: #fff;
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
        }

        .brand-title {
            font-size: 1.3rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #fff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-badges {
            display: flex;
            gap: 0.5rem;
            margin-left: 1rem;
        }

        .badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }

        .badge.cyan {
            background: rgba(6, 182, 212, 0.15);
            color: #67e8f9;
            border-color: rgba(6, 182, 212, 0.3);
        }

        .badge.green {
            background: rgba(16, 185, 129, 0.15);
            color: #6ee7b7;
            border-color: rgba(16, 185, 129, 0.3);
        }

        .container {
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
            padding: 1.5rem 2rem;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 1.5rem;
            flex: 1;
        }

        .panel {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(16px);
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .panel-title {
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .control-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .control-value {
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }

        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            box-shadow: 0 0 10px var(--primary);
            transition: transform 0.1s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        select, button, textarea {
            font-family: inherit;
        }

        select {
            background: rgba(11, 15, 25, 0.8);
            border: 1px solid var(--card-border);
            color: var(--text);
            padding: 0.6rem 0.8rem;
            border-radius: 8px;
            outline: none;
            font-size: 0.9rem;
            cursor: pointer;
        }

        select:focus {
            border-color: var(--primary);
        }

        .presets {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text);
            padding: 0.6rem 0.8rem;
            border-radius: 8px;
            text-align: left;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .preset-btn:hover {
            background: rgba(99, 102, 241, 0.15);
            border-color: var(--primary);
            transform: translateX(4px);
        }

        .meta-card {
            background: rgba(8, 12, 22, 0.6);
            border-radius: 8px;
            padding: 0.8rem;
            font-size: 0.8rem;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
        }

        .meta-row {
            display: flex;
            justify-content: space-between;
        }

        .meta-row span:last-child {
            color: var(--text);
        }

        .main-area {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .prompt-box {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        textarea {
            width: 100%;
            height: 110px;
            background: var(--code-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1rem;
            color: var(--text);
            font-size: 0.95rem;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
        }

        .action-row {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--primary-hover));
            color: #fff;
            border: none;
            padding: 0.75rem 1.8rem;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
            transition: all 0.2s;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }

        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            padding: 0.75rem 1.2rem;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-secondary:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.1);
        }

        .output-card {
            flex: 1;
            min-height: 320px;
            background: var(--code-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            position: relative;
        }

        .output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 0.6rem;
        }

        .output-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .output-stats {
            display: flex;
            gap: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--accent-cyan);
        }

        .output-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            flex: 1;
            overflow-y: auto;
        }

        .prompt-highlight {
            color: #818cf8;
            font-weight: 600;
        }

        .completion-highlight {
            color: #34d399;
        }

        .cursor {
            display: inline-block;
            width: 8px;
            height: 16px;
            background-color: var(--accent);
            animation: blink 0.8s infinite;
            vertical-align: middle;
            margin-left: 2px;
        }

        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }

        .copy-btn {
            background: transparent;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
        }

        .copy-btn:hover {
            color: #fff;
            border-color: #fff;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-logo">N</div>
            <div>
                <div class="brand-title">Nool-Alpha-100M Playground</div>
            </div>
            <div class="brand-badges">
                <span class="badge">GSLA Attention</span>
                <span class="badge cyan">HFK-MoE Top-2</span>
                <span class="badge green" id="headerDevice">CPU Mode</span>
            </div>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-muted);">
            Local Inference Engine (Port 7860)
        </div>
    </header>

    <div class="container">
        <!-- Sidebar Controls -->
        <aside class="panel">
            <div class="panel-title">⚙️ Checkpoint & Tuning</div>

            <div class="control-group">
                <label style="font-size: 0.85rem; font-weight: 500;">Active Checkpoint</label>
                <select id="checkpointSelect" onchange="switchCheckpoint()">
                    <option value="">Loading checkpoints...</option>
                </select>
            </div>

            <div class="control-group">
                <div class="control-header">
                    <span>Temperature</span>
                    <span class="control-value" id="valTemp">0.60</span>
                </div>
                <input type="range" id="paramTemp" min="0.1" max="1.5" step="0.05" value="0.60" oninput="updateVal('valTemp', this.value)">
            </div>

            <div class="control-group">
                <div class="control-header">
                    <span>Top-P (Nucleus)</span>
                    <span class="control-value" id="valTopP">0.85</span>
                </div>
                <input type="range" id="paramTopP" min="0.5" max="1.0" step="0.05" value="0.85" oninput="updateVal('valTopP', this.value)">
            </div>

            <div class="control-group">
                <div class="control-header">
                    <span>Repetition Penalty</span>
                    <span class="control-value" id="valRep">1.20</span>
                </div>
                <input type="range" id="paramRep" min="1.0" max="2.0" step="0.05" value="1.20" oninput="updateVal('valRep', this.value)">
            </div>

            <div class="control-group">
                <div class="control-header">
                    <span>Max New Tokens</span>
                    <span class="control-value" id="valTokens">60</span>
                </div>
                <input type="range" id="paramTokens" min="10" max="200" step="5" value="60" oninput="updateVal('valTokens', this.value)">
            </div>

            <div class="control-group">
                <label style="font-size: 0.85rem; font-weight: 500; margin-bottom: 0.2rem;">Multi-Domain Presets</label>
                <div class="presets">
                    <button class="preset-btn" onclick="applyPreset(1)">
                        <span>🇬🇧 English Reasoning</span>
                        <span>&rarr;</span>
                    </button>
                    <button class="preset-btn" onclick="applyPreset(2)">
                        <span>🇮🇩 Bahasa Indonesia</span>
                        <span>&rarr;</span>
                    </button>
                    <button class="preset-btn" onclick="applyPreset(3)">
                        <span>💻 Python Code</span>
                        <span>&rarr;</span>
                    </button>
                </div>
            </div>

            <div class="meta-card">
                <div class="meta-row"><span>Total Params:</span><span id="metaTotalParams">111.2M</span></div>
                <div class="meta-row"><span>Active Params:</span><span id="metaActiveParams">97.9M</span></div>
                <div class="meta-row"><span>Training Step:</span><span id="metaStep">-</span></div>
                <div class="meta-row"><span>Recorded Loss:</span><span id="metaLoss">-</span></div>
            </div>
        </aside>

        <!-- Main Generation Area -->
        <main class="main-area">
            <div class="panel prompt-box">
                <div class="panel-title">✏️ Input Prompt</div>
                <textarea id="promptInput" placeholder="Ketik prompt dalam Bahasa Indonesia, English, atau kode Python di sini..."></textarea>
                <div class="action-row">
                    <button class="btn-primary" id="btnGenerate" onclick="startGeneration()">
                        <span id="btnIcon">⚡</span>
                        <span id="btnText">Generate Completion</span>
                    </button>
                    <button class="btn-secondary" onclick="clearAll()">Clear</button>
                </div>
            </div>

            <div class="output-card">
                <div class="output-header">
                    <span class="output-title">Real-Time Generated Output</span>
                    <div style="display: flex; align-items: center; gap: 0.8rem;">
                        <div class="output-stats">
                            <span id="statSpeed">- tok/s</span>
                            <span>&bull;</span>
                            <span id="statTime">0.0s</span>
                            <span>&bull;</span>
                            <span id="statCount">0 tokens</span>
                        </div>
                        <button class="copy-btn" onclick="copyOutput()">Copy</button>
                    </div>
                </div>
                <div class="output-text" id="outputContainer">
                    <span class="prompt-highlight" id="renderedPrompt"></span><span class="completion-highlight" id="renderedCompletion">Prompt Anda akan muncul di sini beserta hasil kelanjutan teks dari model...</span><span class="cursor" id="typingCursor" style="display: none;"></span>
                </div>
            </div>
        </main>
    </div>

    <script>
        const PRESETS = {
            1: "Artificial intelligence will transform the future of",
            2: "Ibu kota Nusantara (IKN) merupakan pusat pemerintahan baru",
            3: "def quick_sort(arr):\\n    # Implement quicksort in python\\n"
        };

        let isGenerating = false;
        let eventSource = null;

        function updateVal(elementId, val) {
            document.getElementById(elementId).innerText = parseFloat(val).toFixed(2);
            if (elementId === 'valTokens') {
                document.getElementById(elementId).innerText = val;
            }
        }

        function applyPreset(id) {
            document.getElementById('promptInput').value = PRESETS[id];
        }

        function clearAll() {
            document.getElementById('promptInput').value = '';
            document.getElementById('renderedPrompt').innerText = '';
            document.getElementById('renderedCompletion').innerText = 'Menunggu prompt...';
            document.getElementById('statSpeed').innerText = '- tok/s';
            document.getElementById('statTime').innerText = '0.0s';
            document.getElementById('statCount').innerText = '0 tokens';
        }

        function copyOutput() {
            const p = document.getElementById('renderedPrompt').innerText;
            const c = document.getElementById('renderedCompletion').innerText;
            navigator.clipboard.writeText(p + c);
            alert('Teks berhasil disalin ke clipboard!');
        }

        async function fetchInfo() {
            try {
                const res = await fetch('/api/info');
                const data = await res.json();
                
                // Populate checkpoint selector
                const select = document.getElementById('checkpointSelect');
                select.innerHTML = '';
                data.available_checkpoints.forEach(ckpt => {
                    const opt = document.createElement('option');
                    opt.value = ckpt.filename;
                    opt.innerText = `${ckpt.filename} (${ckpt.size_mb} MB)`;
                    if (ckpt.filename === data.current_checkpoint) {
                        opt.selected = true;
                    }
                    select.appendChild(opt);
                });

                // Metadata
                document.getElementById('metaStep').innerText = data.metadata.step || 'N/A';
                document.getElementById('metaLoss').innerText = data.metadata.loss ? parseFloat(data.metadata.loss).toFixed(4) : 'N/A';
                document.getElementById('metaTotalParams').innerText = (data.metadata.total_params_m || 111.2) + 'M';
                document.getElementById('metaActiveParams').innerText = (data.metadata.active_params_m || 97.9) + 'M';
                document.getElementById('headerDevice').innerText = data.device.toUpperCase() + ' Mode';
            } catch (err) {
                console.error('Failed to load server info:', err);
            }
        }

        async function switchCheckpoint() {
            const select = document.getElementById('checkpointSelect');
            const target = select.value;
            select.disabled = true;
            try {
                const res = await fetch('/api/switch_checkpoint', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ checkpoint: target })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    fetchInfo();
                } else {
                    alert('Gagal memuat checkpoint: ' + data.error);
                }
            } catch (e) {
                alert('Error switching checkpoint: ' + e);
            } finally {
                select.disabled = false;
            }
        }

        async function startGeneration() {
            const prompt = document.getElementById('promptInput').value.trim();
            if (!prompt) {
                alert('Silakan masukkan prompt terlebih dahulu!');
                return;
            }

            if (isGenerating) {
                if (eventSource) eventSource.close();
                isGenerating = false;
                document.getElementById('btnText').innerText = 'Generate Completion';
                document.getElementById('typingCursor').style.display = 'none';
                return;
            }

            const temp = parseFloat(document.getElementById('paramTemp').value);
            const topP = parseFloat(document.getElementById('paramTopP').value);
            const rep = parseFloat(document.getElementById('paramRep').value);
            const maxTokens = parseInt(document.getElementById('paramTokens').value);

            // Reset UI
            document.getElementById('renderedPrompt').innerText = prompt;
            document.getElementById('renderedCompletion').innerText = '';
            document.getElementById('typingCursor').style.display = 'inline-block';
            document.getElementById('btnText').innerText = 'Stop Generation';
            isGenerating = true;

            const url = `/api/stream?prompt=${encodeURIComponent(prompt)}&max_tokens=${maxTokens}&temp=${temp}&top_p=${topP}&rep=${rep}`;
            eventSource = new EventSource(url);

            eventSource.onmessage = function(e) {
                const data = JSON.parse(e.data);
                if (data.accumulated_text !== undefined) {
                    document.getElementById('renderedCompletion').innerText = data.accumulated_text;
                    document.getElementById('statSpeed').innerText = `${data.tok_per_sec} tok/s`;
                    document.getElementById('statTime').innerText = `${data.elapsed_sec}s`;
                    document.getElementById('statCount').innerText = `${data.token_idx + 1} tokens`;
                }

                if (data.is_finished) {
                    eventSource.close();
                    isGenerating = false;
                    document.getElementById('btnText').innerText = 'Generate Completion';
                    document.getElementById('typingCursor').style.display = 'none';
                }
            };

            eventSource.onerror = function() {
                eventSource.close();
                isGenerating = false;
                document.getElementById('btnText').innerText = 'Generate Completion';
                document.getElementById('typingCursor').style.display = 'none';
            };
        }

        window.onload = fetchInfo;
    </script>
</body>
</html>
"""


class PlaygroundRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress verbose standard HTTP request logs
        return

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ["/", "/index.html"]:
            body = HTML_CONTENT.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/info":
            with engine_lock:
                ckpts = inference_engine.get_available_checkpoints()
                info = {
                    "current_checkpoint": inference_engine.current_checkpoint_name,
                    "available_checkpoints": ckpts,
                    "metadata": inference_engine.checkpoint_metadata,
                    "device": str(inference_engine.device),
                }
            self._send_json(200, info)
            return

        if parsed.path == "/api/stream":
            qs = parse_qs(parsed.query)
            prompt = qs.get("prompt", [""])[0]
            max_tokens = int(qs.get("max_tokens", [50])[0])
            temperature = float(qs.get("temp", [0.6])[0])
            top_p = float(qs.get("top_p", [0.85])[0])
            repetition_penalty = float(qs.get("rep", [1.2])[0])

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with engine_lock:
                try:
                    for chunk in inference_engine.stream_generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    ):
                        data_line = f"data: {json.dumps(chunk)}\\n\\n"
                        self.wfile.write(data_line.encode("utf-8"))
                        self.wfile.flush()
                except Exception as e:
                    err_line = f"data: {json.dumps({'error': str(e), 'is_finished': True})}\\n\\n"
                    self.wfile.write(err_line.encode("utf-8"))
                    self.wfile.flush()
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/switch_checkpoint":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                target = payload.get("checkpoint")
                with engine_lock:
                    inference_engine.load_checkpoint(target)
                self._send_json(200, {"status": "success", "active": inference_engine.current_checkpoint_name})
            except Exception as e:
                self._send_json(500, {"status": "error", "error": str(e)})
            return

        self.send_response(404)
        self.end_headers()


def start_web_server(host: str = "127.0.0.1", port: int = 7860, checkpoint_path: Optional[str] = None):
    global inference_engine
    print("[+] Initializing NoolAlphaInference engine...", flush=True)
    inference_engine = NoolAlphaInference(checkpoint_path=checkpoint_path)

    server = None
    for p in [port, port + 1, port + 2]:
        try:
            server = HTTPServer((host, p), PlaygroundRequestHandler)
            port = p
            break
        except OSError:
            continue

    if server is None:
        raise RuntimeError(f"Could not bind HTTP server to {host}:{port}-{port+2}")

    print("=" * 60, flush=True)
    print("[+] Nool-Alpha-100M Interactive Web Playground Active!", flush=True)
    print(f"[+] Local URL:   http://{host}:{port}", flush=True)
    print(f"[+] Checkpoint:  {inference_engine.current_checkpoint_name}", flush=True)
    print(f"[+] Hardware:    {inference_engine.device}", flush=True)
    print("=" * 60, flush=True)
    print("[*] Press Ctrl+C in terminal to stop the server.", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Shutting down server...", flush=True)
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nool-Alpha-100M Web Playground.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="HTTP server port (default: 7860)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Initial checkpoint path")
    args = parser.parse_args()

    start_web_server(host=args.host, port=args.port, checkpoint_path=args.checkpoint)
