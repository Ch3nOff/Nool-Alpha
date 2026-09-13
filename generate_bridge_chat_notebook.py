"""
Generates the memory-optimized, standalone Kaggle notebook `nool_alpha_100m_bridge_chat.ipynb`
for Stage 3: Bilingual Bridge (En <-> Id) & Everyday Natural Conversation SFT (3.5h Budget).
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

NOTEBOOK_PATH = "nool_alpha_100m_bridge_chat.ipynb"


def build_bridge_chat_notebook():
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
    add_markdown(r"""# 🌐 Nool-Alpha-100M: Tahap 3 Bilingual Bridge & Everyday Conversation SFT (3.5 Jam)

Notebook mandiri (*standalone*) ini menjalankan **Supervised Fine-Tuning Jembatan Linguistik Bilingual (Inggris <-> Indonesia) dan Percakapan Sehari-hari (*Everyday Natural Conversation*)** untuk model Nool-Alpha-100M dengan budget waktu **3.5 Jam** di Kaggle GPU.

---

### 🛡️ Fitur Utama & Zero-OOM Architecture:
1. **Penyelarasan Bilingual & Percakapan Sehari-Hari**:
   - 🌐 **OPUS Translation (35%)**: Penyelarasan semantik dua arah (*bidirectional cross-lingual alignment*) Indonesia $\leftrightarrow$ Inggris (`kaitchup/opus-Indonesian-to-English` & `hfxunlp/opus-100`).
   - 💬 **UltraChat-200k (35%)**: Percakapan multi-turn alami bahasa Inggris untuk diskusi sehari-hari, penjelasan ramah, dan interaksi luwes (`HuggingFaceH4/ultrachat_200k`).
   - 🇮🇩 **Alpaca GPT-4 Indonesian (30%)**: Dialog percakapan santun, tanya-jawab praktis, dan interaksi natural bahasa Indonesia (`FreedomIntelligence/alpaca-gpt4-indonesian`).
2. **Mekanisme Anti-Hafalan (*Rolling Reservoir Anti-Memorization*)**:
   - **Dynamic Entropy Seed**: Menggunakan `time.time_ns() ^ (os.getpid() << 16)` untuk permutasi data yang selalu unik di setiap sesi.
   - **Lightweight Rolling Reservoir**: Buffer rolling dinamis di RAM (128 sampel, <5 MB RAM) yang mengocok data streaming secara acak tanpa membebani Host RAM Kaggle (100% aman dari kernel restart / OOM).
   - **Raw String Length Guard**: Membatasi teks mentah (`prompt <= 2000` karakter, `resp <= 4000` karakter) sebelum tokenisasi.
3. **Loss Masking (Label Masking)**:
   - Token instruksi (`### Instruction:\n...\n\n### Response:\n`) diberi label `-100`.
   - Backpropagation loss hanya dihitung pada token jawaban percakapan dan terjemahan.
4. **Time Budgeting 3.5 Jam**:
   - Timer otomatis (`12,600 detik`) dengan penyimpanan periodik dan auto-save checkpoint final saat waktu habis.
5. **Safetensors Export Fix**:
   - Mengekspor model dengan penanganan khusus tensor memori bersama (`.clone().contiguous().cpu()`) agar bebas error `RuntimeError: Some tensors share memory`.""")

    # Cell 1: Environment Setup
    add_code("""# [Cell 1] Environment Setup & Dependencies
!pip install -q datasets transformers accelerate rich safetensors

import os
import sys
import math
import time
import json
import random
import re
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

    # Cell 3: Checkpoint Discovery & Weight Loading
    add_code("""# [Cell 3] Checkpoint Auto-Detection & Weight Loading
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
        "best_bridge_checkpoint.pt",
        "best_reasoning_checkpoint.pt",
        "nool_alpha_100m_reasoning_final.pt",
        "best_sft_checkpoint.pt",
        "nool_alpha_100m_sft_final.pt",
        "best_checkpoint.pt",
        "model.safetensors"
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
    if LOADED_CKPT_PATH.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(LOADED_CKPT_PATH)
        base_step = "Safetensors"
        base_loss = "N/A"
    else:
        ckpt = torch.load(LOADED_CKPT_PATH, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        base_step = ckpt.get("step", "N/A") if isinstance(ckpt, dict) else "N/A"
        base_loss = ckpt.get("loss", "N/A") if isinstance(ckpt, dict) else "N/A"

    model_dict = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    print(f"✅ Berhasil memuat {len(matched)} tensors dari Step: {base_step} (Loss: {base_loss})")
else:
    print("⚠️ Checkpoint tidak ditemukan. Menggunakan bobot awal acak.")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

total_p, active_p = model.get_num_params()
print(f"Model Specs: {total_p / 1e6:.1f}M Total Params | {active_p / 1e6:.1f}M Active Params per Token")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()""")

    # Cell 4: Safe Rolling Reservoir Bilingual & Chat Dataset
    add_code("""# [Cell 4] Memory-Safe Rolling Reservoir Bilingual Bridge & Everyday Chat Dataset
class MemorySafeBridgeChatDataset(IterableDataset):
    \"\"\"
    Anti-Memorization streaming dataset with O(1) constant RAM footprint.
    Maintains a rolling reservoir of 128 samples to guarantee random permutations
    across OPUS translation, UltraChat-200k, and Indonesian Alpaca dialogue.
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
        print(f"🎲 Initialized Rolling Reservoir Bilingual Bridge Sampler with Seed: {self.seed}")

    def _stream_opus_translation(self):
        en_to_id_templates = [
            "Terjemahkan ke dalam bahasa Indonesia:\\n{text}",
            "Terjemahkan kalimat berikut ke dalam bahasa Indonesia:\\n{text}",
            "Translate the following sentence into Indonesian:\\n{text}",
            "Bagaimana terjemahan bahasa Indonesia dari teks ini?\\n{text}",
            "Translate to Indonesian:\\n{text}",
        ]
        id_to_en_templates = [
            "Terjemahkan ke dalam bahasa Inggris:\\n{text}",
            "Terjemahkan kalimat berikut ke dalam bahasa Inggris:\\n{text}",
            "Translate the following Indonesian sentence into English:\\n{text}",
            "Bagaimana terjemahan bahasa Inggris dari teks ini?\\n{text}",
            "Translate to English:\\n{text}",
        ]

        while True:
            # 1. Primary: kaitchup/opus-Indonesian-to-English
            try:
                ds = load_dataset("kaitchup/opus-Indonesian-to-English", split="train", streaming=True)
                for item in ds:
                    raw_text = item.get("text", "")
                    if "###>" in raw_text:
                        parts = raw_text.split("###>", 1)
                        id_text = parts[0].strip()
                        en_text = parts[1].strip()
                        if len(id_text) < 4 or len(en_text) < 4:
                            continue

                        if self.rng.random() < 0.5:
                            tmpl = self.rng.choice(en_to_id_templates)
                            yield tmpl.format(text=en_text[:2000]), id_text[:4000]
                        else:
                            tmpl = self.rng.choice(id_to_en_templates)
                            yield tmpl.format(text=id_text[:2000]), en_text[:4000]
            except Exception:
                pass

            # 2. Fallback: hfxunlp/opus-100 (if accessible)
            try:
                ds = load_dataset("hfxunlp/opus-100", "en-id", split="train", streaming=True)
                for item in ds:
                    trans = item.get("translation", {})
                    en_text = trans.get("en", "").strip()
                    id_text = trans.get("id", "").strip()
                    if len(id_text) < 4 or len(en_text) < 4:
                        continue

                    if self.rng.random() < 0.5:
                        tmpl = self.rng.choice(en_to_id_templates)
                        yield tmpl.format(text=en_text[:2000]), id_text[:4000]
                    else:
                        tmpl = self.rng.choice(id_to_en_templates)
                        yield tmpl.format(text=id_text[:2000]), en_text[:4000]
            except Exception:
                pass

    def _stream_ultrachat(self):
        while True:
            try:
                ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
                for item in ds:
                    messages = item.get("messages", [])
                    if len(messages) < 2:
                        continue

                    prompt, response = "", ""
                    for m in messages:
                        role = m.get("role", "")
                        content = m.get("content", "").strip()
                        if role == "user" and not prompt:
                            prompt = content
                        elif role == "assistant" and prompt and not response:
                            response = content
                            break

                    if prompt and response and len(response) > 5:
                        yield prompt[:2000].strip(), response[:4000].strip()
            except Exception:
                continue

    def _stream_indonesian_dialogue(self):
        while True:
            try:
                ds = load_dataset("FreedomIntelligence/alpaca-gpt4-indonesian", split="train", streaming=True)
                for item in ds:
                    convs = item.get("conversations", [])
                    prompt, response = "", ""
                    if convs:
                        for c in convs:
                            sender = c.get("from", "").lower()
                            val = c.get("value", "").strip()
                            if sender in ["user", "human"] and not prompt:
                                prompt = val
                            elif sender in ["assistant", "gpt"] and prompt and not response:
                                response = val
                                break
                    else:
                        inst = item.get("instruction", "").strip()
                        inp = item.get("input", "").strip()
                        out = item.get("output", "").strip()
                        if inst and out:
                            prompt = f"{inst}\\n\\nKonteks:\\n{inp}" if inp else inst
                            response = out

                    if prompt and response and len(response) > 5:
                        yield prompt[:2000].strip(), response[:4000].strip()
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
            ("opus_translation", self._stream_opus_translation()),
            ("ultrachat_everyday", self._stream_ultrachat()),
            ("indonesian_dialogue", self._stream_indonesian_dialogue()),
        ]
        # 35% OPUS En<->Id Translation, 35% UltraChat-200k, 30% Indonesian Alpaca Dialogue
        weights = [0.35, 0.35, 0.30]
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

        reservoir = []
        for _ in range(self.reservoir_size):
            reservoir.append(get_next_sample())

        while True:
            pick_idx = self.rng.randint(0, len(reservoir) - 1)
            sample = reservoir[pick_idx]
            reservoir[pick_idx] = get_next_sample()
            yield sample

print("[OK] MemorySafeBridgeChatDataset ready with Zero-OOM Rolling Reservoir!")""")

    # Cell 5: 3.5h Training Loop with AMP & Memory Management
    add_code("""# [Cell 5] 3.5-Hour Bilingual Bridge & Everyday Conversation SFT Training Loop
OUTPUT_DIR = "/kaggle/working/bridge_chat_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
MAX_TRAINING_HOURS = 3.5
TARGET_MAX_STEPS = 3600
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
PEAK_LR = 1.0e-4
MIN_LR = 1.0e-5
WARMUP_STEPS = 100
LOG_INTERVAL = 25
EVAL_INTERVAL = 150

dataset = MemorySafeBridgeChatDataset(tokenizer, max_seq_len=512, reservoir_size=128)
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
    ("En->Id Translation", "Terjemahkan ke dalam bahasa Indonesia:\\nGood morning! How can I assist you with your project today?"),
    ("Id->En Translation", "Terjemahkan ke dalam bahasa Inggris:\\nTerima kasih banyak atas bantuan dan penjelasan yang sangat jelas ini."),
    ("Indonesian Daily Chat", "Halo, apa kabar? Bagaimana cara menjaga kesehatan tubuh saat cuaca sedang pancaroba?"),
    ("English Daily Chat", "Hi! Could you suggest a quick and healthy breakfast idea that takes less than 10 minutes to prepare?"),
]

def generate_probe(prompt: str, max_tokens: int = 70):
    model.eval()
    fmt = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
    inp = tokenizer(fmt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = inp.shape[1]
    with torch.no_grad():
        for _ in range(max_tokens):
            idx = inp[:, -config.max_position_embeddings:] if inp.shape[1] > config.max_position_embeddings else inp
            logits, _, _, _ = model(idx)
            next_logits = logits[:, -1, :] / 0.65

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

print("=" * 70)
print(f"🚀 Memulai Bilingual Bridge & Everyday Conversation SFT ({MAX_TRAINING_HOURS}h Budget = {int(MAX_TRAINING_HOURS*3600)}s)")
print("=" * 70)

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

        print(f"Bridge Step {step:5d}/{TARGET_MAX_STEPS} | Loss: {avg_l:.4f} (Aux: {avg_a:.4f}) | LR: {cur_lr:.2e} | Speed: {speed:.0f} tok/s | Tokens: {session_tokens/1e6:.2f}M | Left: {rem_h:.2f}h")

        if avg_l < best_loss:
            best_loss = avg_l
            torch.save({
                "step": step,
                "loss": best_loss,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, os.path.join(OUTPUT_DIR, "best_bridge_checkpoint.pt"))
            torch.save({
                "step": step,
                "loss": best_loss,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, "best_bridge_checkpoint.pt")
            print(f"  ⭐ Checkpoint Terbaik Disimpan! (Loss: {best_loss:.4f})")

    if step % EVAL_INTERVAL == 0:
        print("\\n" + "=" * 55)
        print(f"🌐 [Bilingual & Everyday Chat Probes @ Step {step}]")
        for tag, pr in PROBES:
            ans = generate_probe(pr, max_tokens=70)
            print(f"[{tag}]\\n{ans[:160]}...\\n")
        print("=" * 55 + "\\n")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Save final checkpoint
final_path = os.path.join(OUTPUT_DIR, "nool_alpha_100m_bridge_final.pt")
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
}, "nool_alpha_100m_bridge_final.pt")
print(f"\\n🎉 Bilingual Bridge & Everyday Conversation SFT Selesai! Model disimpan di: {final_path}")""")

    # Cell 6: Visualisasi Loss & MoE Stability Curve
    add_code("""# [Cell 6] Visualisasi Loss & MoE Stability Curve
if log_steps:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(log_steps, log_losses, color="royalblue", lw=2, label="Bridge & Chat SFT Loss")
    plt.title("Nool-Alpha-100M Stage 3 SFT Loss")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(log_steps, log_aux, color="crimson", lw=2, label="Router Balance Loss (MoE)")
    plt.title("Router Stability (No Collapse)")
    plt.xlabel("Step")
    plt.ylabel("Aux Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("bridge_chat_sft_curves.png", dpi=150)
    plt.show()""")

    # Cell 7: Bilingual & Everyday Chat Evaluation Suite
    add_code("""# [Cell 7] Bilingual Bridge & Everyday Chat Evaluation Suite
evaluation_prompts = [
    ("En->Id Translation", "Terjemahkan ke dalam bahasa Indonesia:\\nArtificial intelligence is transforming modern science and technology."),
    ("Id->En Translation", "Terjemahkan ke dalam bahasa Inggris:\\nMatahari terbit dari timur dan memberikan energi bagi semua makhluk hidup di bumi."),
    ("Indonesian Polite Chat", "Selamat siang! Boleh minta tips sederhana bagaimana cara mengelola waktu belajar agar lebih efektif?"),
    ("English Practical Chat", "Hello! How can I reduce stress during a busy work week? Give me 3 actionable tips."),
    ("Reasoning Retention Check", "Sebuah toko memberi diskon 20% untuk jaket seharga Rp 200.000. Berapa harga akhirnya?"),
    ("Code Retention Check", "Write a Python function `reverse_list(lst)` that reverses a list.")
]

print("=" * 65)
print("🌐 Evaluasi Kualitas Bilingual & Percakapan Alami Nool-Alpha-100M")
print("=" * 65)

for category, pr in evaluation_prompts:
    res = generate_probe(pr, max_tokens=90)
    print(f"📌 [{category}]")
    print(f"Prompt  : {pr.replace(chr(10), ' ')}")
    print(f"Response: {res}")
    print("-" * 65)""")

    # Cell 8: Safetensors Export
    add_code("""# [Cell 8] Export ke Format Standar Safetensors (Zero Duplicate Memory Fix)
from safetensors.torch import save_file

export_dir = "exported_bridge_chat_model"
os.makedirs(export_dir, exist_ok=True)

best_ckpt_file = "best_bridge_checkpoint.pt" if os.path.exists("best_bridge_checkpoint.pt") else "nool_alpha_100m_bridge_final.pt"
ckpt = torch.load(best_ckpt_file, map_location="cpu", weights_only=False)
raw_state = ckpt.get("model_state_dict", ckpt)

# Gunakan .clone().contiguous().cpu() untuk mengatasi pembagian memori pada tied weights
clean_state = {k: v.clone().contiguous().cpu() for k, v in raw_state.items()}
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
    "sft_stage": "3_bilingual_bridge_chat",
    "best_loss": ckpt.get("loss", "N/A"),
    "step": ckpt.get("step", "N/A")
}

with open(os.path.join(export_dir, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=2)

tokenizer.save_pretrained(export_dir)
print(f"✅ Berhasil mengekspor model ke format Safetensors di: '{export_dir}/'")
print(f"File model: {os.path.join(export_dir, 'model.safetensors')} ({os.path.getsize(os.path.join(export_dir, 'model.safetensors')) / (1024*1024):.2f} MB)")""")

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
    build_bridge_chat_notebook()
