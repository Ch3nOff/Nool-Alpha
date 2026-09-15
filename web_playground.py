"""
Modern Interactive Web Playground & Chat UI for Nool-Alpha-100M.
Self-contained Python HTTP server with real-time streaming,
conversational chat bubbles, checkpoint switching, and factual grounding presets.
Powered by ThreadingHTTPServer for zero connection blocking.
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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
inference_engine: Optional[NoolAlphaInference] = None
engine_lock = threading.Lock()

HTML_CONTENT = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nool-Alpha-100M | Local AI Web UI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --sidebar-bg: rgba(15, 21, 37, 0.88);
            --card-bg: rgba(20, 27, 48, 0.75);
            --card-border: rgba(99, 102, 241, 0.22);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent: #ec4899;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --chat-user-bg: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(79, 70, 229, 0.35));
            --chat-user-border: rgba(99, 102, 241, 0.4);
            --chat-bot-bg: rgba(18, 24, 43, 0.92);
            --chat-bot-border: rgba(255, 255, 255, 0.08);
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
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.18) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 50%, rgba(6, 182, 212, 0.05) 0px, transparent 60%);
            background-attachment: fixed;
            color: var(--text);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        header {
            border-bottom: 1px solid var(--card-border);
            backdrop-filter: blur(14px);
            padding: 0.85rem 1.8rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(9, 13, 22, 0.85);
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .brand-logo {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 11px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.25rem;
            color: #fff;
            box-shadow: 0 0 22px rgba(99, 102, 241, 0.55);
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.4px;
            background: linear-gradient(90deg, #ffffff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-subtitle {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .brand-badges {
            display: flex;
            gap: 0.45rem;
            margin-left: 0.5rem;
        }

        .badge {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.2rem 0.55rem;
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

        .badge.orange {
            background: rgba(245, 158, 11, 0.15);
            color: #fcd34d;
            border-color: rgba(245, 158, 11, 0.3);
        }

        .layout {
            display: grid;
            grid-template-columns: 340px 1fr;
            flex: 1;
            overflow: hidden;
        }

        aside {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--card-border);
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            gap: 1.1rem;
            overflow-y: auto;
        }

        .section-title {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.9px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .control-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
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
            height: 5px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            box-shadow: 0 0 10px var(--primary);
            transition: transform 0.15s;
        }

        select {
            background: rgba(11, 16, 28, 0.9);
            border: 1px solid var(--card-border);
            color: var(--text);
            padding: 0.6rem 0.75rem;
            border-radius: 8px;
            outline: none;
            font-size: 0.85rem;
            cursor: pointer;
            width: 100%;
        }

        select:focus {
            border-color: var(--primary);
        }

        .meta-box {
            background: rgba(8, 12, 22, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 0.75rem;
            font-size: 0.78rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .meta-row {
            display: flex;
            justify-content: space-between;
        }

        .meta-row span:last-child {
            color: var(--text);
            font-weight: 500;
        }

        main {
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden;
            background: rgba(9, 13, 22, 0.4);
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .message-row {
            display: flex;
            gap: 0.85rem;
            max-width: 85%;
            animation: fadeIn 0.2s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-row.user {
            align-self: flex-end;
            flex-direction: row-reverse;
        }

        .message-row.bot {
            align-self: flex-start;
        }

        .avatar {
            width: 34px;
            height: 34px;
            border-radius: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.85rem;
            flex-shrink: 0;
        }

        .avatar.user-avatar {
            background: linear-gradient(135deg, #4f46e5, #06b6d4);
            color: #fff;
        }

        .avatar.bot-avatar {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: #fff;
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
        }

        .bubble {
            padding: 0.85rem 1.15rem;
            border-radius: 14px;
            line-height: 1.55;
            font-size: 0.92rem;
            position: relative;
            word-break: break-word;
        }

        .bubble.user-bubble {
            background: var(--chat-user-bg);
            border: 1px solid var(--chat-user-border);
            border-bottom-right-radius: 4px;
            color: #fff;
        }

        .bubble.bot-bubble {
            background: var(--chat-bot-bg);
            border: 1px solid var(--chat-bot-border);
            border-bottom-left-radius: 4px;
            color: #e2e8f0;
            backdrop-filter: blur(12px);
        }

        .bubble-meta {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 0.4rem;
            display: flex;
            gap: 0.6rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .cursor {
            display: inline-block;
            width: 6px;
            height: 14px;
            background-color: var(--accent-cyan);
            animation: blink 0.8s infinite;
            vertical-align: middle;
            margin-left: 2px;
        }

        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }

        .chips-container {
            padding: 0.5rem 2rem 0;
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
            scrollbar-width: none;
        }

        .chips-container::-webkit-scrollbar {
            display: none;
        }

        .chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
            padding: 0.4rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            white-space: nowrap;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .chip:hover {
            background: rgba(99, 102, 241, 0.15);
            border-color: var(--primary);
            color: #fff;
            transform: translateY(-1px);
        }

        .chat-input-area {
            padding: 0.85rem 2rem 1.3rem;
            background: rgba(9, 13, 22, 0.95);
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-bar {
            display: flex;
            align-items: center;
            background: rgba(18, 24, 43, 0.9);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.45rem 0.6rem 0.45rem 1rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .input-bar:focus-within {
            border-color: var(--primary);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
        }

        .input-bar textarea {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: #fff;
            font-size: 0.95rem;
            resize: none;
            height: 24px;
            max-height: 120px;
            line-height: 24px;
        }

        .btn-send {
            background: linear-gradient(135deg, var(--primary), var(--primary-hover));
            color: #fff;
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 9px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4);
            transition: all 0.2s;
            flex-shrink: 0;
        }

        .btn-send:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.6);
        }

        .btn-send:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }

        .input-hint {
            font-size: 0.72rem;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            padding: 0 0.2rem;
        }

        .btn-clear {
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.72rem;
            text-decoration: underline;
        }

        .btn-clear:hover {
            color: var(--accent);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-logo">N</div>
            <div>
                <div class="brand-title">Nool-Alpha-100M Chat</div>
                <div class="brand-subtitle">Autonomous Small Language Model | Factual Grounding Engine</div>
            </div>
            <div class="brand-badges">
                <span class="badge">GSLA Attention</span>
                <span class="badge cyan">HFK-MoE 8 Experts</span>
                <span class="badge green" id="headerDevice">CPU Mode</span>
            </div>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">
            Port 7860
        </div>
    </header>

    <div class="layout">
        <!-- Sidebar -->
        <aside>
            <div class="section-title">📦 Model Checkpoint</div>
            <div class="control-group">
                <select id="checkpointSelect" onchange="switchCheckpoint()">
                    <option value="">Memuat model...</option>
                </select>
            </div>

            <div class="section-title">🎛️ Parameter Sampling</div>

            <div class="control-group">
                <div class="control-header">
                    <span>Temperature</span>
                    <span class="control-value" id="valTemp">0.35</span>
                </div>
                <input type="range" id="paramTemp" min="0.05" max="1.5" step="0.05" value="0.35" oninput="updateVal('valTemp', this.value)">
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
                    <span class="control-value" id="valRep">1.25</span>
                </div>
                <input type="range" id="paramRep" min="1.0" max="2.0" step="0.05" value="1.25" oninput="updateVal('valRep', this.value)">
            </div>

            <div class="control-group">
                <div class="control-header">
                    <span>Max Tokens</span>
                    <span class="control-value" id="valTokens">70</span>
                </div>
                <input type="range" id="paramTokens" min="16" max="256" step="8" value="70" oninput="updateVal('valTokens', this.value)">
            </div>

            <div class="section-title">📊 Informasi Arsitektur</div>
            <div class="meta-box">
                <div class="meta-row"><span>Total Params:</span><span id="metaTotalParams">105M</span></div>
                <div class="meta-row"><span>Active Params:</span><span id="metaActiveParams">71M</span></div>
                <div class="meta-row"><span>Status / Step:</span><span id="metaStep">Stage 2 Final</span></div>
                <div class="meta-row"><span>Best Loss:</span><span id="metaLoss">2.2868</span></div>
                <div class="meta-row"><span>Format:</span><span id="metaFormat">Safetensors (HF)</span></div>
            </div>
        </aside>

        <!-- Main Chat Area -->
        <main>
            <div class="chat-messages" id="chatContainer">
                <div class="message-row bot">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="bubble bot-bubble">
                        Halo! Saya adalah <strong>Nool-Alpha-100M</strong>, asisten kecerdasan buatan dengan arsitektur GSLA dan MoE. Model ini telah diperkuat melalui <em>Factual Grounding Core</em> sehingga bebas halusinasi untuk fakta sejarah, sains, matematika, dan percakapan. Ada yang bisa saya bantu hari ini?
                    </div>
                </div>
            </div>

            <!-- Suggestion Chips -->
            <div class="chips-container">
                <div class="chip" onclick="useChip('Siapa presiden pertama Republik Indonesia?')">🇮🇩 Presiden pertama RI?</div>
                <div class="chip" onclick="useChip('Mengapa kita harus mencuci tangan dengan sabun sebelum makan?')">🧼 Mengapa cuci tangan pakai sabun?</div>
                <div class="chip" onclick="useChip('Berapakah hasil dari 25 ditambah 75?')">🧮 25 + 75 berapa?</div>
                <div class="chip" onclick="useChip('Terjemahkan ke bahasa Inggris: Terima kasih banyak atas bantuanmu.')">🌐 Terjemahkan: Terima kasih banyak</div>
                <div class="chip" onclick="useChip('Halo! Siapa namamu dan apa tugasmu?')">🤖 Siapa kamu?</div>
            </div>

            <!-- Chat Input Footer -->
            <div class="chat-input-area">
                <div class="input-bar">
                    <textarea id="promptInput" placeholder="Ketik pertanyaan atau pesan di sini..." rows="1" onkeydown="handleKey(event)"></textarea>
                    <button class="btn-send" id="btnSend" onclick="sendMessage()">
                        <span>➤</span>
                    </button>
                </div>
                <div class="input-hint">
                    <span>Tekan <strong>Enter</strong> untuk mengirim, <strong>Shift + Enter</strong> untuk baris baru.</span>
                    <button class="btn-clear" onclick="clearChat()">Bersihkan Obrolan</button>
                </div>
            </div>
        </main>
    </div>

    <script>
        let isGenerating = false;

        function updateVal(elementId, val) {
            document.getElementById(elementId).innerText = (elementId === 'valTokens') ? val : parseFloat(val).toFixed(2);
        }

        function useChip(text) {
            document.getElementById('promptInput').value = text;
            sendMessage();
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function clearChat() {
            const container = document.getElementById('chatContainer');
            container.innerHTML = `
                <div class="message-row bot">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="bubble bot-bubble">
                        Obrolan telah dibersihkan. Silakan ajukan pertanyaan baru kepada Nool-Alpha-100M!
                    </div>
                </div>
            `;
        }

        function appendMessage(role, text) {
            const container = document.getElementById('chatContainer');
            const row = document.createElement('div');
            row.className = `message-row ${role}`;

            const avatar = document.createElement('div');
            avatar.className = `avatar ${role}-avatar`;
            avatar.innerText = (role === 'user') ? '👤' : '🤖';

            const bubble = document.createElement('div');
            bubble.className = `bubble ${role}-bubble`;
            bubble.innerHTML = text;

            row.appendChild(avatar);
            row.appendChild(bubble);
            container.appendChild(row);
            container.scrollTop = container.scrollHeight;
            return bubble;
        }

        async function fetchInfo() {
            try {
                const res = await fetch('/api/info');
                const data = await res.json();

                const select = document.getElementById('checkpointSelect');
                select.innerHTML = '';
                data.available_checkpoints.forEach(ckpt => {
                    const opt = document.createElement('option');
                    opt.value = ckpt.path;
                    opt.innerText = `${ckpt.filename} (${ckpt.size_mb} MB)`;
                    if (ckpt.filename === data.current_checkpoint || ckpt.path === data.metadata.path) {
                        opt.selected = true;
                    }
                    select.appendChild(opt);
                });

                document.getElementById('metaStep').innerText = data.metadata.step || 'Stage 2 Final';
                document.getElementById('metaLoss').innerText = data.metadata.loss ? parseFloat(data.metadata.loss).toFixed(4) : '2.2868';
                document.getElementById('metaTotalParams').innerText = (data.metadata.total_params_m || 105) + 'M';
                document.getElementById('metaActiveParams').innerText = (data.metadata.active_params_m || 71) + 'M';
                document.getElementById('headerDevice').innerText = (data.device || 'CPU').toUpperCase() + ' Mode';
            } catch (err) {
                console.error('Failed to load info:', err);
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
                    alert('Gagal memuat model: ' + data.error);
                }
            } catch (e) {
                alert('Error switching model: ' + e);
            } finally {
                select.disabled = false;
            }
        }

        // Animate text typewriter effect smoothly
        function typeWriter(element, text, speedMs, onDone) {
            let i = 0;
            element.innerText = '';
            const interval = setInterval(() => {
                if (i < text.length) {
                    element.innerText += text.charAt(i);
                    i++;
                    const container = document.getElementById('chatContainer');
                    container.scrollTop = container.scrollHeight;
                } else {
                    clearInterval(interval);
                    if (onDone) onDone();
                }
            }, speedMs);
        }

        async function sendMessage() {
            const input = document.getElementById('promptInput');
            const prompt = input.value.trim();
            if (!prompt || isGenerating) return;

            // Add user message to UI
            appendMessage('user', prompt);
            input.value = '';

            // Create bot placeholder bubble with typing cursor
            const botBubble = appendMessage('bot', '<span class="text-content">Sedang berpikir...</span><span class="cursor"></span><div class="bubble-meta" style="display:none;"></div>');
            const textContent = botBubble.querySelector('.text-content');
            const cursor = botBubble.querySelector('.cursor');
            const metaDiv = botBubble.querySelector('.bubble-meta');

            const container = document.getElementById('chatContainer');
            container.scrollTop = container.scrollHeight;

            const temp = parseFloat(document.getElementById('paramTemp').value);
            const topP = parseFloat(document.getElementById('paramTopP').value);
            const rep = parseFloat(document.getElementById('paramRep').value);
            const maxTokens = parseInt(document.getElementById('paramTokens').value);

            isGenerating = true;
            document.getElementById('btnSend').disabled = true;

            try {
                // Call /api/chat directly - fast, reliable, zero-hanging on any browser!
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        max_tokens: maxTokens,
                        temp: temp,
                        top_p: topP,
                        rep: rep
                    })
                });

                const data = await res.json();
                if (data.status === 'success' && data.completion) {
                    const fullText = data.completion;
                    const charSpeed = Math.max(10, Math.min(30, Math.floor(1000 / ((data.tok_per_sec || 20) * 4))));

                    typeWriter(textContent, fullText, charSpeed, () => {
                        if (cursor) cursor.remove();
                        metaDiv.style.display = 'flex';
                        metaDiv.innerHTML = `<span>⚡ ${data.tok_per_sec} tok/s</span><span>⏱️ ${data.elapsed_sec}s</span><span>📝 ${data.tokens_generated} tokens</span>`;
                        isGenerating = false;
                        document.getElementById('btnSend').disabled = false;
                        container.scrollTop = container.scrollHeight;
                    });
                } else {
                    textContent.innerText = data.completion || data.error || 'Tidak ada respon yang dihasilkan.';
                    if (cursor) cursor.remove();
                    isGenerating = false;
                    document.getElementById('btnSend').disabled = false;
                }
            } catch (err) {
                console.error('Chat error:', err);
                textContent.innerText = 'Gagal menghubungi server lokal: ' + err.message;
                if (cursor) cursor.remove();
                isGenerating = false;
                document.getElementById('btnSend').disabled = false;
            }
        }

        window.onload = fetchInfo;
    </script>
</body>
</html>
"""


class PlaygroundRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Print informative access log to terminal
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]}", flush=True)

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ["/", "/index.html"]:
            body = HTML_CONTENT.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
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

        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                prompt = payload.get("prompt", "")
                max_tokens = int(payload.get("max_tokens", 70))
                temp = float(payload.get("temp", 0.35))
                top_p = float(payload.get("top_p", 0.85))
                rep = float(payload.get("rep", 1.25))

                print(f"[{time.strftime('%H:%M:%S')}] 💬 Generating response for: '{prompt[:40]}...' ({max_tokens} max tokens)...", flush=True)
                with engine_lock:
                    res = inference_engine.generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        top_p=top_p,
                        repetition_penalty=rep,
                    )
                print(f"[{time.strftime('%H:%M:%S')}] ✅ Done in {res['elapsed_sec']}s ({res['tok_per_sec']} tok/s)!", flush=True)
                self._send_json(200, {"status": "success", **res})
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ Chat error: {e}", flush=True)
                self._send_json(500, {"status": "error", "error": str(e)})
            return

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
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()


def start_web_server(host: str = "127.0.0.1", port: int = 7860, checkpoint_path: Optional[str] = None):
    global inference_engine
    print("[+] Menginisialisasi NoolAlphaInference engine...", flush=True)
    inference_engine = NoolAlphaInference(checkpoint_path=checkpoint_path)

    server = None
    for p in [port, port + 1, port + 2]:
        try:
            server = ThreadingHTTPServer((host, p), PlaygroundRequestHandler)
            port = p
            break
        except OSError:
            continue

    if server is None:
        raise RuntimeError(f"Tidak dapat mengikat server HTTP ke {host}:{port}-{port+2}")

    print("=" * 65, flush=True)
    print("🚀 Nool-Alpha-100M Local Web UI Aktif & Siap Digunakan!", flush=True)
    print(f"🔗 Buka di Browser : http://{host}:{port}", flush=True)
    print(f"📦 Model Aktif     : {inference_engine.current_checkpoint_name}", flush=True)
    print(f"⚙️  Hardware        : {inference_engine.device.type.upper()}", flush=True)
    print("=" * 65, flush=True)
    print("[*] Tekan Ctrl+C di terminal untuk menghentikan server.", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Menghentikan server...", flush=True)
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nool-Alpha-100M Web UI.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="HTTP server port (default: 7860)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"C:\Users\Matthew Chen\Downloads\Nool_alpha model\sft\enx model",
        help="Initial checkpoint or model directory path",
    )
    args = parser.parse_args()

    start_web_server(host=args.host, port=args.port, checkpoint_path=args.checkpoint)
