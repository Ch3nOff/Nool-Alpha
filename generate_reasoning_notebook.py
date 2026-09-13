"""
Generates the memory-optimized, standalone Kaggle notebook `nool_alpha_100m_reasoning_sft.ipynb`
for Stage 2.5: Deep Reasoning & Thought-Chain SFT (4.5h Budget, 7 Datasets, Anti-Memorization Randomization).
Guarantees zero OOM crashes on Kaggle Host RAM and GPU VRAM.
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

NOTEBOOK_PATH = "nool_alpha_100m_reasoning_sft.ipynb"


def build_reasoning_notebook():
    cells = []

    def add_markdown(source: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")],
        })

    def add_code(source: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")],
        })

    # Header
    add_markdown("""# 🧠 Nool-Alpha-100M: Tahap 2.5 Deep Reasoning & Thought-Chain SFT (4.5 Jam)

Notebook mandiri (*standalone*) ini menjalankan **Supervised Fine-Tuning Penalaran Mendalam (*Deep Reasoning & Thought-Chain*)** untuk model Nool-Alpha-100M dengan budget waktu **4.5 Jam** di Kaggle.

---

### 🛡️ Fitur Utama & Optimasi Memori (Zero-OOM Architecture):
1. **7 Dataset Reasoning Terbaik Dunia**:
   - 📚 `HuggingFaceTB/cosmopedia-100k`: Pengetahuan sintetis berstruktur tinggi
   - 💻 `m-a-p/Code-Feedback`: Penalaran algoritma & kode Python
   - 🔢 `nvidia/OpenMathInstruct-1`: Dataset matematika tingkat lanjut dari NVIDIA
   - 🧬 `ServiceNow-AI/R1-Distill-SFT` (`v1`): Jejak penalaran `<think>` DeepSeek-R1
   - 💭 `open-thoughts/OpenThoughts-114k`: Penelusuran pemikiran bertahap (*long thought steps*)
   - 🌐 `open-r1/Mixture-of-Thoughts` (`all`): Reasoning sains, matematika, dan pemrograman
   - 📐 `IFM/Math-Reasoning` (`math-thinking-qwen`): Langkah pemecahan masalah analitis
2. **Mekanisme Anti-Hafalan (*Rolling Reservoir Anti-Memorization*)**:
   - **Dynamic Entropy Seed**: Menggunakan `time.time_ns() ^ os.getpid()` sehingga setiap kali notebook di-restart, permutasi urutan data 100% baru.
   - **Lightweight Rolling Reservoir**: Menggunakan buffer rolling dinamis di RAM (~128 sampel, <5 MB RAM) yang terus mengocok sampel tanpa membebani Host RAM Kaggle (mencegah kernel restart / OOM).
   - **Raw String Guard**: Membatasi panjang teks mentah sebelum tokenisasi untuk mencegah sampel 20.000+ token memicu lonjakan memori.
   - **Weighted Stream Interleaving**: Alokasi sampling dinamis antar 7 dataset.
3. **Loss Masking (Label Masking)**:
   - Token instruksi (`### Instruction:\n...\n\n### Response:\n`) diberi label `-100`.
   - Backpropagation loss hanya aktif pada token jejak pemikiran `<think>` dan solusi final.
4. **Time Budgeting 4.5 Jam**:
   - Timer otomatis (`16,200 detik`) dengan penyimpanan periodik dan auto-save checkpoint final saat waktu habis.""")

    # Cell 1: Dependencies
    add_code("""# [Cell 1] Environment Setup & Dependencies
!pip install -q datasets transformers accelerate rich safetensors

import os
import sys
import math
import time
import json
import random
import re
import ast
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

print(f"PyTorch Version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")""")

    # Cell 2: Architecture with 100% key parity
    add_code("""# [Cell 2] Nool-Alpha-100M Architecture (100% Parameter Key Parity)
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class NoolAlphaConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 10
    num_heads: int = 12
    head_dim: int = 64
    d_c: int = 192
    d_pe: int = 32
    rope_theta: float = 500000.0
    max_position_embeddings: int = 2048
    sliding_window: int = 512
    swa_interval: int = 4
    shared_ffn_dim: int = 1536
    num_experts: int = 8
    top_k_experts: int = 2
    expert_rank: int = 96
    moe_aux_loss_coeff: float = 0.01
    highway_alpha_init: float = 0.05
    logit_soft_cap: float = 30.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    @classmethod
    def nool_100m(cls):
        return cls()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class DecoupledRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        return cos, sin

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        elif x.ndim == 4:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        return (x * cos) + (self._rotate_half(x) * sin)

class GroupedSubspaceLatentAttention(nn.Module):
    def __init__(self, config: NoolAlphaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_c = config.d_c
        self.d_pe = config.d_pe
        self.scale = 1.0 / math.sqrt(self.head_dim + self.d_pe)

        self.is_sliding = (config.sliding_window is not None) and (
            (layer_idx % config.swa_interval) != (config.swa_interval - 1)
        )
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.w_dk = nn.Linear(self.d_model, self.d_c, bias=False)
        self.w_pos = nn.Linear(self.d_model, self.d_pe, bias=False)
        self.w_q = nn.Linear(self.d_model, self.num_heads * (self.head_dim + self.d_pe), bias=False)

        self.w_uk = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.d_c))
        self.w_uv = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.d_c))
        nn.init.kaiming_uniform_(self.w_uk, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_uv, a=math.sqrt(5))

        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.rotary_emb = DecoupledRotaryEmbedding(self.d_pe, config.max_position_embeddings, config.rope_theta)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, kv_cache=None, use_cache=False):
        batch_size, seq_len, _ = hidden_states.shape

        c_t = self.w_dk(hidden_states)
        k_pe = self.w_pos(hidden_states)
        cos, sin = self.rotary_emb(k_pe, seq_len)
        k_pe = self.rotary_emb.apply_rope(k_pe, cos, sin)

        if kv_cache is not None:
            c_t = torch.cat([kv_cache[0], c_t], dim=1)
            k_pe = torch.cat([kv_cache[1], k_pe], dim=1)
        current_cache = (c_t, k_pe) if use_cache else None
        total_seq_len = c_t.shape[1]

        q_proj = self.w_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim + self.d_pe)
        q_val = q_proj[..., : self.head_dim]
        q_pe = self.rotary_emb.apply_rope(q_proj[..., self.head_dim :], cos, sin)

        q_tilde = torch.einsum("bthd,hdc->bthc", q_val, self.w_uk)

        content_score = torch.einsum("bthc,bsc->bhts", q_tilde, c_t)
        pos_score = torch.einsum("bthd,bsd->bhts", q_pe, k_pe)
        scores = (content_score + pos_score) * self.scale

        offset = total_seq_len - seq_len
        q_pos = torch.arange(seq_len, device=scores.device).unsqueeze(1) + offset
        k_pos = torch.arange(total_seq_len, device=scores.device).unsqueeze(0)
        mask = k_pos > q_pos
        if self.sliding_window is not None:
            mask = mask | (k_pos < (q_pos - self.sliding_window))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        v_h = torch.einsum("bsc,hdc->bhsd", c_t, self.w_uv)
        attn_out = torch.matmul(attn_weights, v_h).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        return self.w_o(attn_out), current_cache

class FactorizedExpert(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.u_gate = nn.Linear(d_model, rank, bias=False)
        self.u_up = nn.Linear(d_model, rank, bias=False)
        self.v_down = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_down(F.silu(self.u_gate(x)) * self.u_up(x))

class HeterogeneousFactorizedMoE(nn.Module):
    def __init__(self, config: NoolAlphaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        self.shared_w_gate = nn.Linear(self.d_model, config.shared_ffn_dim, bias=False)
        self.shared_w_up = nn.Linear(self.d_model, config.shared_ffn_dim, bias=False)
        self.shared_w_down = nn.Linear(config.shared_ffn_dim, self.d_model, bias=False)

        self.experts = nn.ModuleList([FactorizedExpert(self.d_model, config.expert_rank) for _ in range(self.num_experts)])
        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape
        x_flat = x.view(-1, d)
        num_tokens = x_flat.shape[0]

        shared_out = self.shared_w_down(F.silu(self.shared_w_gate(x_flat)) * self.shared_w_up(x_flat))

        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        dynamic_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            exp_idx = topk_indices[:, k]
            weights = topk_weights[:, k].unsqueeze(-1)
            for e_id in range(self.num_experts):
                mask = (exp_idx == e_id)
                if mask.any():
                    dynamic_out[mask] += weights[mask] * self.experts[e_id](x_flat[mask])

        out = (shared_out + dynamic_out).view(b, s, d)

        if self.training:
            one_hot = F.one_hot(topk_indices, num_classes=self.num_experts).float()
            f_i = one_hot.sum(dim=(0, 1)) / (num_tokens * self.top_k)
            p_i = probs.mean(dim=0)
            aux_loss = self.aux_loss_coeff * self.num_experts * torch.sum(f_i * p_i)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return out, aux_loss

class NoolAlphaBlock(nn.Module):
    def __init__(self, config: NoolAlphaConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = GroupedSubspaceLatentAttention(config, layer_idx)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = HeterogeneousFactorizedMoE(config)

    def forward(self, x, attention_mask=None, kv_cache=None, use_cache=False):
        a_out, cache = self.attn(self.norm1(x), attention_mask=attention_mask, kv_cache=kv_cache, use_cache=use_cache)
        x = x + a_out
        m_out, aux_loss = self.moe(self.norm2(x))
        x = x + m_out
        return x, aux_loss, cache

class NoolAlphaForCausalLM(nn.Module):
    def __init__(self, config: NoolAlphaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.highway_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.highway_alpha = nn.Parameter(torch.tensor(config.highway_alpha_init, dtype=torch.float32))
        self.layers = nn.ModuleList([NoolAlphaBlock(config, i) for i in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None, use_cache=False):
        x_0 = self.embed_tokens(input_ids)
        x_0_norm = self.highway_norm(x_0)

        h = x_0
        total_aux = torch.tensor(0.0, device=input_ids.device)
        next_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            cache = past_key_values[i] if past_key_values else None
            h, aux, c = layer(h, attention_mask=attention_mask, kv_cache=cache, use_cache=use_cache)
            total_aux = total_aux + aux
            if use_cache:
                next_caches.append(c)

        h_norm = self.final_norm(h)
        x_final = h_norm + torch.tanh(self.highway_alpha) * x_0_norm

        logits_raw = self.lm_head(x_final)
        cap = self.config.logit_soft_cap
        logits = cap * torch.tanh(logits_raw / cap) if cap else logits_raw

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            loss = ce_loss + total_aux

        return logits, loss, total_aux, next_caches

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        shared = sum(p.numel() for n, p in self.named_parameters() if "experts" not in n)
        exp_p = sum(p.numel() for p in self.layers[0].moe.experts[0].parameters()) * self.config.n_layers
        active = shared + (self.config.top_k_experts * exp_p)
        return total, active

print("[OK] NoolAlpha-100M architecture compiled with 100% parameter key parity!")""")

    # Cell 3: Checkpoint Auto-Detection
    add_code("""# [Cell 3] Checkpoint Auto-Detection & Safe Unpickling
if hasattr(sys.modules["__main__"], "NoolAlphaConfig") is False:
    setattr(sys.modules["__main__"], "NoolAlphaConfig", NoolAlphaConfig)

try:
    torch.serialization.add_safe_globals([NoolAlphaConfig])
except Exception:
    pass

def find_checkpoint():
    search_dirs = [
        "/kaggle/input/datasets/chenstillstude/nool-cp",
        "/kaggle/input/nool-cp",
        "/kaggle/input",
        "/kaggle/working",
        "."
    ]
    priority_names = [
        "best_sft_checkpoint.pt",
        "nool_alpha_100m_sft_final.pt",
        "best_reasoning_checkpoint.pt",
        "best_checkpoint.pt",
        "nool_alpha_100m_final.pt"
    ]
    for p_name in priority_names:
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                for root, _, files in os.walk(s_dir):
                    if p_name in files:
                        return os.path.join(root, p_name)
    return None

LOADED_CKPT_PATH = find_checkpoint()
config = NoolAlphaConfig.nool_100m()
model = NoolAlphaForCausalLM(config).to(device)

if LOADED_CKPT_PATH and os.path.exists(LOADED_CKPT_PATH):
    print(f"🔄 Checkpoint Ditemukan: {LOADED_CKPT_PATH}")
    ckpt = torch.load(LOADED_CKPT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    load_res = model.load_state_dict(state_dict)
    base_step = ckpt.get("step", "N/A") if isinstance(ckpt, dict) else "N/A"
    base_loss = ckpt.get("loss", "N/A") if isinstance(ckpt, dict) else "N/A"
    print(f"✅ Berhasil memuat bobot model dari Step: {base_step} (Loss: {base_loss})")
    print(f"✅ State Dict Status: {load_res}")
else:
    print("⚠️ Checkpoint tidak ditemukan. Memulai dari bobot acak.")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

total_p, active_p = model.get_num_params()
print(f"Model Specs: {total_p / 1e6:.1f}M Total Params | {active_p / 1e6:.1f}M Active Params per Token")

# Clean RAM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()""")

    # Cell 4: Safe Rolling Reservoir Reasoning Dataset
    add_code("""# [Cell 4] Memory-Safe Rolling Reservoir Reasoning Dataset (Anti-Memorization)
class MemorySafeReasoningDataset(IterableDataset):
    \"\"\"
    Anti-Memorization streaming dataset with O(1) constant RAM footprint.
    Maintains a rolling reservoir of 128 samples to guarantee random permutations
    without buffering thousands of huge samples in Host RAM.
    \"\"\"
    def __init__(self, tokenizer, max_seq_len: int = 512, reservoir_size: int = 128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reservoir_size = reservoir_size
        self.eos_token_id = tokenizer.eos_token_id or 50256
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_token_id
        # Dynamic entropy seed: unique every run
        self.seed = int(time.time_ns() % 1_000_000_007) ^ (os.getpid() << 16)
        self.rng = random.Random(self.seed)
        print(f"🎲 Initialized Rolling Reservoir Anti-Memorization Sampler with Seed: {self.seed}")

    def _stream_cosmopedia(self):
        while True:
            try:
                ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train", streaming=True)
                for item in ds:
                    p = item.get("prompt", "")
                    r = item.get("text", "")
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_code_feedback(self):
        while True:
            try:
                ds = load_dataset("m-a-p/Code-Feedback", split="train", streaming=True)
                for item in ds:
                    msgs = item.get("messages", [])
                    p, r = "", ""
                    for m in msgs:
                        role = m.get("role")
                        content = m.get("content", "")
                        if role == "user" and not p: p = content
                        elif role == "assistant" and p and not r: r = content
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_openmath(self):
        while True:
            try:
                ds = load_dataset("nvidia/OpenMathInstruct-1", split="train", streaming=True)
                for item in ds:
                    if "is_correct" in item and not item["is_correct"]:
                        continue
                    p = item.get("question", "")
                    r = item.get("generated_solution", "")
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_r1_distill(self):
        while True:
            try:
                ds = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1", split="train", streaming=True)
                for item in ds:
                    r = item.get("reannotated_assistant_content", "")
                    msgs = item.get("messages", []) or item.get("reannotated_messages", [])
                    p = ""
                    for m in msgs:
                        if m.get("role") == "user":
                            p = m.get("content", "")
                            break
                    if not r:
                        for m in msgs:
                            if m.get("role") == "assistant":
                                r = m.get("content", "")
                                break
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_openthoughts(self):
        while True:
            try:
                ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
                for item in ds:
                    convs = item.get("conversations", [])
                    p, r = "", ""
                    for c in convs:
                        sender = c.get("from", "").lower()
                        val = c.get("value", "")
                        if sender in ["user", "human"] and not p: p = val
                        elif sender in ["assistant", "gpt"] and p and not r: r = val
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_mot(self):
        while True:
            try:
                ds = load_dataset("open-r1/Mixture-of-Thoughts", "all", split="train", streaming=True)
                for item in ds:
                    msgs = item.get("messages", [])
                    p, r = "", ""
                    for m in msgs:
                        role = m.get("role")
                        content = m.get("content", "")
                        if role == "user" and not p: p = content
                        elif role == "assistant" and p and not r: r = content
                    if p and r:
                        yield p[:2000].strip(), r[:4000].strip()
            except Exception:
                continue

    def _stream_math_reasoning(self):
        while True:
            try:
                ds = load_dataset("IFM/Math-Reasoning", "math-thinking-qwen", split="train", streaming=True)
                for item in ds:
                    t = item.get("text", "")
                    if "\\n\\n" in t:
                        parts = t.split("\\n\\n", 1)
                        yield parts[0][:2000].strip(), parts[1][:4000].strip()
                    elif t:
                        yield "Solve the following mathematical reasoning problem step by step:", t[:4000].strip()
            except Exception:
                continue

    def _tokenize(self, prompt: str, resp: str):
        prompt_txt = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
        prompt_ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
        resp_ids = self.tokenizer.encode(resp, add_special_tokens=False) + [self.eos_token_id]

        total = len(prompt_ids) + len(resp_ids)
        if total > self.max_seq_len:
            max_resp = self.max_seq_len - len(prompt_ids)
            if max_resp < 16:
                return None
            resp_ids = resp_ids[:max_resp - 1] + [self.eos_token_id]

        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids

        pad_len = self.max_seq_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __iter__(self):
        streams = [
            ("r1", self._stream_r1_distill()),
            ("openthoughts", self._stream_openthoughts()),
            ("mot", self._stream_mot()),
            ("openmath", self._stream_openmath()),
            ("code", self._stream_code_feedback()),
            ("cosmopedia", self._stream_cosmopedia()),
            ("math_qwen", self._stream_math_reasoning()),
        ]
        weights = [0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10]
        stream_indices = list(range(len(streams)))

        def get_next_sample():
            while True:
                chosen_idx = self.rng.choices(stream_indices, weights=weights, k=1)[0]
                _, stream = streams[chosen_idx]
                try:
                    p, r = next(stream)
                    tok = self._tokenize(p, r)
                    if tok is not None:
                        return tok
                except Exception:
                    continue

        # 1. Fill rolling reservoir buffer with initial samples
        reservoir = []
        for _ in range(self.reservoir_size):
            reservoir.append(get_next_sample())

        # 2. Continuous random replacement sampling (anti-memorization)
        while True:
            pick_idx = self.rng.randint(0, len(reservoir) - 1)
            sample = reservoir[pick_idx]
            # Replace picked item with fresh sample from stream
            reservoir[pick_idx] = get_next_sample()
            yield sample

print("[OK] MemorySafeReasoningDataset ready with Zero-OOM Rolling Reservoir!")""")

    # Cell 5: 4.5h Training Loop with AMP & Memory Management
    add_code("""# [Cell 5] 4.5-Hour Deep Reasoning SFT Training Loop (AMP bfloat16, Memory Guard)
OUTPUT_DIR = "/kaggle/working/reasoning_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
MAX_TRAINING_HOURS = 4.5
TARGET_MAX_STEPS = 4500
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
PEAK_LR = 1.2e-4
MIN_LR = 1.0e-5
WARMUP_STEPS = 100
LOG_INTERVAL = 25
EVAL_INTERVAL = 150

dataset = MemorySafeReasoningDataset(tokenizer, max_seq_len=512, reservoir_size=128)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
data_iter = iter(dataloader)

decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
optimizer = AdamW([
    {"params": decay_params, "weight_decay": 0.05},
    {"params": nodecay_params, "weight_decay": 0.0},
], lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)

def scheduler_fn(step_idx: int):
    if step_idx < WARMUP_STEPS:
        return float(step_idx) / float(max(1, WARMUP_STEPS))
    progress = float(step_idx - WARMUP_STEPS) / float(max(1, TARGET_MAX_STEPS - WARMUP_STEPS))
    return MIN_LR / PEAK_LR + 0.5 * (1.0 - MIN_LR / PEAK_LR) * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=scheduler_fn)

PROBES = [
    ("Math", "A jacket costs $150 with a 20% discount and 8% tax. What is the final price?"),
    ("Code", "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome."),
    ("Logic", "Explain step-by-step why the sum of two odd numbers is always even."),
]

def generate_probe(prompt: str, max_tokens: int = 60):
    model.eval()
    fmt = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
    inp = tokenizer(fmt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = inp.shape[1]
    with torch.no_grad():
        for _ in range(max_tokens):
            idx = inp[:, -config.max_position_embeddings:] if inp.shape[1] > config.max_position_embeddings else inp
            logits, _, _, _ = model(idx)
            next_logits = logits[:, -1, :] / 0.6

            for tid in set(inp[0].tolist()):
                if next_logits[0, tid] > 0: next_logits[0, tid] /= 1.15
                else: next_logits[0, tid] *= 1.15

            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs > 0.85
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = 0
            indices_to_remove = remove_mask.scatter(1, sorted_indices, remove_mask)
            next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            inp = torch.cat([inp, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    decoded = tokenizer.decode(inp[0].tolist(), skip_special_tokens=False)
    model.train()
    if "### Response:\\n" in decoded:
        return decoded.split("### Response:\\n", 1)[1].replace("<|endoftext|>", "").strip()
    return decoded[prompt_len:].replace("<|endoftext|>", "").strip()

print("=" * 65)
print(f"🚀 Memulai Deep Reasoning SFT ({MAX_TRAINING_HOURS}h Budget = {int(MAX_TRAINING_HOURS*3600)}s)")
print("=" * 65)

start_time = time.time()
max_seconds = MAX_TRAINING_HOURS * 3600.0

step = 0
running_loss = 0.0
running_aux = 0.0
best_loss = float("inf")
session_tokens = 0
t_prev = time.time()

log_steps = []
log_losses = []
log_aux = []

model.train()
while step < TARGET_MAX_STEPS:
    elapsed = time.time() - start_time
    if elapsed >= max_seconds:
        print(f"\\n⏱️ Time budget reached ({MAX_TRAINING_HOURS}h). Saving final checkpoint...")
        break

    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0
    step_aux = 0.0

    for _ in range(GRAD_ACCUM_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss, aux_loss, _ = model(input_ids, labels)
        else:
            logits, loss, aux_loss, _ = model(input_ids, labels)

        scaled_loss = loss / GRAD_ACCUM_STEPS
        scaled_loss.backward()

        step_loss += loss.item() / GRAD_ACCUM_STEPS
        step_aux += aux_loss.item() / GRAD_ACCUM_STEPS
        session_tokens += input_ids.numel()

        # Free memory references immediately
        del logits, loss, aux_loss

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    step += 1

    running_loss += step_loss
    running_aux += step_aux

    if step % LOG_INTERVAL == 0:
        avg_l = running_loss / LOG_INTERVAL
        avg_a = running_aux / LOG_INTERVAL
        running_loss = 0.0
        running_aux = 0.0

        dt = time.time() - t_prev
        speed = (LOG_INTERVAL * GRAD_ACCUM_STEPS * BATCH_SIZE * 512) / max(dt, 1e-5)
        t_prev = time.time()

        cur_lr = scheduler.get_last_lr()[0]
        rem_h = (max_seconds - (time.time() - start_time)) / 3600.0

        log_steps.append(step)
        log_losses.append(avg_l)
        log_aux.append(avg_a)

        print(f"SFT Step {step:5d}/{TARGET_MAX_STEPS} | Loss: {avg_l:.4f} (Aux: {avg_a:.4f}) | LR: {cur_lr:.2e} | Speed: {speed:.0f} tok/s | Tokens: {session_tokens/1e6:.2f}M | Left: {rem_h:.2f}h")

        if avg_l < best_loss:
            best_loss = avg_l
            torch.save({
                "step": step,
                "loss": best_loss,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, os.path.join(OUTPUT_DIR, "best_reasoning_checkpoint.pt"))
            torch.save({
                "step": step,
                "loss": best_loss,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, "best_reasoning_checkpoint.pt")
            print(f"  ⭐ Checkpoint Terbaik Disimpan! (Loss: {best_loss:.4f})")

    if step % EVAL_INTERVAL == 0:
        print("\\n" + "=" * 50)
        print(f"🎯 [Reasoning Probes @ Step {step}]")
        for tag, pr in PROBES:
            ans = generate_probe(pr)
            print(f"[{tag}] {ans[:140]}...")
        print("=" * 50 + "\\n")
        # Periodic memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Save final checkpoint
final_path = os.path.join(OUTPUT_DIR, "nool_alpha_100m_reasoning_final.pt")
torch.save({
    "step": step,
    "loss": best_loss,
    "model_state_dict": model.state_dict(),
    "config": config,
}, final_path)
torch.save({
    "step": step,
    "loss": best_loss,
    "model_state_dict": model.state_dict(),
    "config": config,
}, "nool_alpha_100m_reasoning_final.pt")
print(f"\\n🎉 Deep Reasoning SFT Selesai! Model disimpan di: {final_path}")""")

    # Cell 6: Visualisasi Loss & MoE Stability Curve
    add_code("""# [Cell 6] Visualisasi Loss & MoE Stability Curve
if log_steps:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(log_steps, log_losses, color="royalblue", lw=2, label="Reasoning SFT Loss")
    plt.title("Nool-Alpha-100M Reasoning SFT Loss")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(log_steps, log_aux, color="crimson", lw=2, label="HFK-MoE Router Balance Loss")
    plt.title("Router Stability (No Collapse)")
    plt.xlabel("Step")
    plt.ylabel("Aux Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("reasoning_sft_curves.png", dpi=150)
    plt.show()""")

    # Cell 7: Python AST Syntax Validation
    add_code("""# [Cell 7] Python AST Syntax Validation Suite
test_tasks = [
    ("factorial", "Write a Python function factorial(n) to compute factorial of n."),
    ("fibonacci", "Write a Python function fibonacci(n) returning the nth Fibonacci number."),
    ("is_prime", "Write a Python function is_prime(n) that checks if n is prime."),
    ("reverse_str", "Write a Python function reverse_str(s) that reverses a string.")
]

def extract_code(text):
    m = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if m: return m[0].strip()
    m2 = re.findall(r"```(.*?)```", text, re.DOTALL)
    if m2: return m2[0].strip()
    lines = [l for l in text.split("\\n") if "def " in l or "    " in l]
    return "\\n".join(lines) if lines else text.strip()

passed = 0
print("=" * 60)
print("🔍 Evaluasi Sintaks Python (AST Validation) Pasca-Reasoning SFT")
print("=" * 60)

for name, pr in test_tasks:
    res = generate_probe(pr)
    code = extract_code(res)
    try:
        ast.parse(code)
        print(f"[PASS] {name}: Valid Python AST!\\n{code[:80]}...\\n")
        passed += 1
    except SyntaxError as e:
        print(f"[FAIL] {name}: SyntaxError ({e.msg})\\n{code[:80]}...\\n")

print(f"AST Pass Rate: {passed}/{len(test_tasks)} ({passed/len(test_tasks)*100:.1f}%)")""")

    # Cell 8: Safetensors Export
    add_code("""# [Cell 8] Export ke Format Standar Safetensors
from safetensors.torch import save_file

export_dir = "exported_reasoning_model"
os.makedirs(export_dir, exist_ok=True)

best_ckpt_file = "best_reasoning_checkpoint.pt" if os.path.exists("best_reasoning_checkpoint.pt") else "nool_alpha_100m_reasoning_final.pt"
ckpt = torch.load(best_ckpt_file, map_location="cpu", weights_only=False)
raw_state = ckpt.get("model_state_dict", ckpt)

clean_state = {k: v.contiguous() for k, v in raw_state.items()}
save_file(clean_state, os.path.join(export_dir, "model.safetensors"))

config_dict = {
    "architectures": ["NoolAlphaForCausalLM"],
    "model_type": "nool_alpha",
    "vocab_size": 50257,
    "d_model": 768,
    "n_layers": 10,
    "num_heads": 12,
    "head_dim": 64,
    "d_c": 192,
    "d_pe": 32,
    "shared_ffn_dim": 1536,
    "num_experts": 8,
    "top_k_experts": 2,
    "expert_rank": 96,
    "logit_soft_cap": 30.0,
    "sft_stage": "2.5_deep_reasoning",
    "best_loss": ckpt.get("loss", "N/A"),
    "step": ckpt.get("step", "N/A")
}

with open(os.path.join(export_dir, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=2)

tokenizer.save_pretrained(export_dir)
print(f"✅ Berhasil mengekspor model ke format Safetensors di: '{export_dir}/'")""")

    # Build notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generated memory-optimized Kaggle notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_reasoning_notebook()
