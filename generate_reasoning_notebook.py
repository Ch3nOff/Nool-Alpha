"""
Generates the complete, standalone Kaggle notebook `nool_alpha_100m_reasoning_sft.ipynb`
for Stage 2.5: Deep Reasoning & Thought-Chain SFT (4.5h Budget, 7 Datasets, Anti-Memorization Randomization).
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

### 🛡️ Fitur Utama Pipeline Ini:
1. **7 Dataset Reasoning Terbaik Dunia**:
   - 📚 `HuggingFaceTB/cosmopedia-100k`: Pengetahuan sintetis berstruktur tinggi
   - 💻 `m-a-p/Code-Feedback`: Penalaran algoritma & kode Python
   - 🔢 `nvidia/OpenMathInstruct-1`: Dataset matematika tingkat lanjut dari NVIDIA
   - 🧬 `ServiceNow-AI/R1-Distill-SFT` (`v1`): Jejak penalaran `<think>` DeepSeek-R1
   - 💭 `open-thoughts/OpenThoughts-114k`: Penelusuran pemikiran bertahap (*long thought steps*)
   - 🌐 `open-r1/Mixture-of-Thoughts` (`all`): Reasoning sains, matematika, dan pemrograman
   - 📐 `IFM/Math-Reasoning` (`math-thinking-qwen`): Langkah pemecahan masalah analitis
2. **Mekanisme Anti-Hafalan (*True Randomized Multi-Stream*)**:
   - **Dynamic Entropy Seed**: Menggunakan `time.time_ns() ^ os.getpid()` sehingga setiap kali notebook di-restart, permutasi urutan data 100% baru.
   - **Shuffle Buffer 10.000 Sampel**: Mencegah model membaca data berurutan dari atas ke bawah.
   - **Random Shard Offset (Skip Acak)**: Melompati 0–3.000 sampel awal saat startup agar langsung mengambil data dari bagian dalam shard.
   - **Weighted Multinomial Sampling**: Alokasi sampling dinamis antar 7 dataset.
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

    # Cell 2: Architecture
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
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q_pe: torch.Tensor, k_pe: torch.Tensor, seq_len: int):
        cos = self.cos_cached[:seq_len, :].to(q_pe.dtype).to(q_pe.device)
        sin = self.sin_cached[:seq_len, :].to(q_pe.dtype).to(q_pe.device)
        q_rot = (q_pe * cos) + (self._rotate_half(q_pe) * sin)
        k_rot = (k_pe * cos) + (self._rotate_half(k_pe) * sin)
        return q_rot, k_rot

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

        self.is_swa = (layer_idx % config.swa_interval != 0)
        self.sliding_window = config.sliding_window if self.is_swa else None

        self.w_q = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.w_q_pe = nn.Linear(self.d_model, self.num_heads * self.d_pe, bias=False)
        self.w_dk = nn.Linear(self.d_model, self.d_c, bias=False)
        self.w_pos = nn.Linear(self.d_model, self.d_pe, bias=False)
        self.w_uk = nn.Linear(self.d_c, self.num_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(self.d_c, self.num_heads * self.head_dim, bias=False)
        self.w_out = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

        self.rotary_emb = DecoupledRotaryEmbedding(
            dim=self.d_pe,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta
        )
        self.scale = 1.0 / math.sqrt(self.head_dim + self.d_pe)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        q_val = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q_pe = self.w_q_pe(x).view(B, T, self.num_heads, self.d_pe).transpose(1, 2)

        c = self.w_dk(x)
        k_pe = self.w_pos(x).view(B, T, 1, self.d_pe).transpose(1, 2)
        k_val = self.w_uk(c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q_pe, k_pe = self.rotary_emb(q_pe, k_pe, seq_len=T)
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)

        q_comb = torch.cat([q_val, q_pe], dim=-1)
        k_comb = torch.cat([k_val, k_pe], dim=-1)

        scores = torch.matmul(q_comb, k_comb.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        if self.sliding_window is not None:
            band_mask = torch.tril(torch.full((T, T), float("-inf"), device=x.device), diagonal=-self.sliding_window)
            causal_mask = causal_mask + band_mask

        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                ext_mask = (1.0 - attention_mask[:, None, None, :].to(scores.dtype)) * -10000.0
                scores = scores + ext_mask

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_out(out)

class FactorizedExpert(nn.Module):
    def __init__(self, d_model: int, expert_rank: int, ffn_dim: int):
        super().__init__()
        self.u_gate = nn.Linear(d_model, expert_rank, bias=False)
        self.v_gate = nn.Linear(expert_rank, ffn_dim, bias=False)
        self.u_up = nn.Linear(d_model, expert_rank, bias=False)
        self.v_up = nn.Linear(expert_rank, ffn_dim, bias=False)
        self.v_down = nn.Linear(ffn_dim, expert_rank, bias=False)
        self.u_down = nn.Linear(expert_rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.v_gate(self.u_gate(x)))
        up = self.v_up(self.u_up(x))
        return self.u_down(self.v_down(gate * up))

class HeterogeneousFactorizedMoE(nn.Module):
    def __init__(self, config: NoolAlphaConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        self.shared_w_gate = nn.Linear(config.d_model, config.shared_ffn_dim, bias=False)
        self.shared_w_up = nn.Linear(config.d_model, config.shared_ffn_dim, bias=False)
        self.shared_w_down = nn.Linear(config.shared_ffn_dim, config.d_model, bias=False)

        self.router = nn.Linear(config.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FactorizedExpert(
                d_model=config.d_model,
                expert_rank=config.expert_rank,
                ffn_dim=config.shared_ffn_dim
            ) for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.shared_w_down(F.silu(self.shared_w_gate(x)) * self.shared_w_up(x))

        B, T, D = x.shape
        flat_x = x.view(-1, D)
        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)

        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        sparse_out = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            k_indices = indices[:, k]
            k_weights = weights[:, k].unsqueeze(-1)
            for e_idx, expert in enumerate(self.experts):
                mask = (k_indices == e_idx)
                if mask.any():
                    tokens = flat_x[mask]
                    exp_out = expert(tokens)
                    sparse_out[mask] += exp_out * k_weights[mask]

        out = shared_out + sparse_out.view(B, T, D)

        density = router_probs.mean(dim=0)
        mask_top = torch.zeros_like(router_probs)
        mask_top.scatter_(1, indices, 1.0)
        route_frac = mask_top.mean(dim=0)
        aux_loss = self.aux_loss_coeff * self.num_experts * torch.sum(density * route_frac)

        return out, aux_loss

class NoolAlphaDecoderLayer(nn.Module):
    def __init__(self, config: NoolAlphaConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.self_attn = GroupedSubspaceLatentAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = HeterogeneousFactorizedMoE(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        residual = x
        x = residual + self.self_attn(self.input_layernorm(x), attention_mask=attention_mask)
        moe_out, aux_loss = self.moe(self.post_attention_layernorm(x))
        x = x + moe_out
        return x, aux_loss

class NoolAlphaForCausalLM(nn.Module):
    def __init__(self, config: NoolAlphaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            NoolAlphaDecoderLayer(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.highway_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.highway_alpha = nn.Parameter(torch.tensor(config.highway_alpha_init))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.logit_soft_cap = config.logit_soft_cap

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x0 = self.embed_tokens(input_ids)
        x = x0

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux = layer(x, attention_mask=attention_mask)
            total_aux_loss = total_aux_loss + aux

        x = self.norm(x)
        highway = torch.tanh(self.highway_alpha) * self.highway_norm(x0)
        x = x + highway

        logits = self.lm_head(x)
        if self.logit_soft_cap > 0:
            logits = self.logit_soft_cap * torch.tanh(logits / self.logit_soft_cap)

        return logits, None, total_aux_loss

print("[OK] NoolAlphaForCausalLM class compiled successfully!")""")

    # Cell 3: Checkpoint auto-loader
    add_code("""# [Cell 3] Auto-Mount Checkpoint (Base or SFT)
# PyTorch 2.6+ safe unpickling registration
if hasattr(sys.modules["__main__"], "NoolAlphaConfig") is False:
    setattr(sys.modules["__main__"], "NoolAlphaConfig", NoolAlphaConfig)

try:
    torch.serialization.add_safe_globals([NoolAlphaConfig])
except Exception:
    pass

def find_checkpoint():
    search_dirs = [
        "/kaggle/input",
        "/kaggle/working",
        "./checkpoints",
        "."
    ]
    # Priority order: latest SFT checkpoint first, then pre-trained base
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
config = NoolAlphaConfig()
model = NoolAlphaForCausalLM(config).to(device)

if LOADED_CKPT_PATH and os.path.exists(LOADED_CKPT_PATH):
    print(f"🔄 Checkpoint Ditemukan: {LOADED_CKPT_PATH}")
    ckpt = torch.load(LOADED_CKPT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_dict = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    base_step = ckpt.get("step", "N/A") if isinstance(ckpt, dict) else "N/A"
    base_loss = ckpt.get("loss", "N/A") if isinstance(ckpt, dict) else "N/A"
    print(f"✅ Sukses Memuat {len(matched)} / {len(model_dict)} Parameter Tensors! (Base Step: {base_step}, Loss: {base_loss})")
else:
    print("⚠️ Checkpoint tidak ditemukan di /kaggle/input/. Memulai dari bobot acak.")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id""")

    # Cell 4: 7-Dataset Streaming with 3-Layer Randomization
    add_code("""# [Cell 4] 7-Dataset Streaming Pipeline with Anti-Memorization 3-Layer Randomization
REASONING_DATASET_CONFIGS = [
    {"tag": "r1_distill", "repo": "ServiceNow-AI/R1-Distill-SFT", "config": "v1", "weight": 0.20},
    {"tag": "openthoughts", "repo": "open-thoughts/OpenThoughts-114k", "config": None, "weight": 0.15},
    {"tag": "mot", "repo": "open-r1/Mixture-of-Thoughts", "config": "all", "weight": 0.15},
    {"tag": "openmath", "repo": "nvidia/OpenMathInstruct-1", "config": None, "weight": 0.15},
    {"tag": "code_feedback", "repo": "m-a-p/Code-Feedback", "config": None, "weight": 0.15},
    {"tag": "cosmopedia", "repo": "HuggingFaceTB/cosmopedia-100k", "config": None, "weight": 0.10},
    {"tag": "math_reasoning", "repo": "IFM/Math-Reasoning", "config": "math-thinking-qwen", "weight": 0.10},
]

def extract_prompt_response(item, tag):
    try:
        if tag == "cosmopedia":
            prompt = item.get("prompt", "").strip()
            response = item.get("text", "").strip()
            if prompt and response: return prompt, response

        elif tag == "code_feedback":
            msgs = item.get("messages", [])
            p, r = "", ""
            for m in msgs:
                if m.get("role") == "user" and not p: p = m.get("content", "").strip()
                elif m.get("role") == "assistant" and p and not r: r = m.get("content", "").strip()
            if p and r: return p, r

        elif tag == "openmath":
            if "is_correct" in item and not item["is_correct"]: return None
            p = item.get("question", "").strip()
            r = item.get("generated_solution", "").strip()
            if p and r: return p, r

        elif tag == "r1_distill":
            r = item.get("reannotated_assistant_content", "").strip()
            msgs = item.get("messages", []) or item.get("reannotated_messages", [])
            p = ""
            for m in msgs:
                if m.get("role") == "user": p = m.get("content", "").strip(); break
            if not r:
                for m in msgs:
                    if m.get("role") == "assistant": r = m.get("content", "").strip(); break
            if p and r: return p, r

        elif tag == "openthoughts":
            convs = item.get("conversations", [])
            p, r = "", ""
            for c in convs:
                sender = c.get("from", "").lower()
                val = c.get("value", "").strip()
                if sender in ["user", "human"] and not p: p = val
                elif sender in ["assistant", "gpt"] and p and not r: r = val
            if p and r: return p, r

        elif tag == "mot":
            msgs = item.get("messages", [])
            p, r = "", ""
            for m in msgs:
                if m.get("role") == "user" and not p: p = m.get("content", "").strip()
                elif m.get("role") == "assistant" and p and not r: r = m.get("content", "").strip()
            if p and r: return p, r

        elif tag == "math_reasoning":
            text = item.get("text", "").strip()
            if "\\n\\n" in text:
                parts = text.split("\\n\\n", 1)
                return parts[0].strip(), parts[1].strip()
            if text:
                return "Solve the following math problem step by step:", text
    except Exception:
        return None
    return None

class KaggleRandomizedReasoningDataset(IterableDataset):
    def __init__(self, tokenizer, max_seq_len=512, buffer_size=10000, max_skip=3000):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.max_skip = max_skip
        # Layer 1: Entropy Dynamic Seed
        self.seed = int(time.time_ns() % 1_000_000_007) ^ (os.getpid() << 16)
        self.rng = random.Random(self.seed)
        self.dataset_configs = REASONING_DATASET_CONFIGS
        self.weights = [c["weight"] for c in self.dataset_configs]
        print(f"🎲 Initialized Anti-Memorization Sampler with Dynamic Seed: {self.seed}")

    def _create_stream(self, cfg, s_seed):
        kwargs = {"split": "train", "streaming": True}
        if cfg["config"]: kwargs["name"] = cfg["config"]
        ds = load_dataset(cfg["repo"], **kwargs)
        # Layer 2: 10,000 Sample Shuffle Buffer
        ds = ds.shuffle(buffer_size=self.buffer_size, seed=s_seed)
        # Layer 3: Random Shard Skip Offset
        if self.max_skip > 0:
            skip_n = self.rng.randint(0, self.max_skip)
            try: ds = ds.skip(skip_n)
            except Exception: pass
        return iter(ds)

    def _tokenize(self, prompt, response):
        prompt_txt = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
        full_txt = f"{prompt_txt}{response}<|endoftext|>"

        p_ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
        f_ids = self.tokenizer.encode(full_txt, add_special_tokens=False)
        if len(p_ids) >= self.max_seq_len - 10: return None
        if len(f_ids) > self.max_seq_len: f_ids = f_ids[:self.max_seq_len]

        input_ids = f_ids.copy()
        labels = f_ids.copy()
        # Loss Masking: Mask prompt tokens with -100
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attn_mask = [1] * len(f_ids) + [0] * pad_len
        else:
            attn_mask = [1] * len(input_ids)

        if sum(1 for l in labels if l != -100) < 4: return None
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __iter__(self):
        streams = []
        active_indices = []
        for i, cfg in enumerate(self.dataset_configs):
            s_seed = (self.seed + i * 7919) % (2**31 - 1)
            try:
                streams.append(self._create_stream(cfg, s_seed))
                active_indices.append(i)
            except Exception as e:
                print(f"[!] Warning stream '{cfg['tag']}': {e}")
                streams.append(None)

        while active_indices:
            cur_w = [self.weights[i] for i in active_indices]
            tot_w = sum(cur_w)
            norm_w = [w / tot_w for w in cur_w]
            chosen = self.rng.choices(active_indices, weights=norm_w, k=1)[0]
            stream = streams[chosen]
            tag = self.dataset_configs[chosen]["tag"]

            try:
                item = next(stream)
            except StopIteration:
                new_s = self.rng.randint(0, 2**31 - 1)
                try:
                    streams[chosen] = self._create_stream(self.dataset_configs[chosen], new_s)
                    item = next(streams[chosen])
                except Exception:
                    active_indices.remove(chosen)
                    continue
            except Exception:
                continue

            extracted = extract_prompt_response(item, tag)
            if extracted is None: continue
            tok = self._tokenize(extracted[0], extracted[1])
            if tok is not None: yield tok

print("[OK] KaggleRandomizedReasoningDataset ready with anti-memorization active!")""")

    # Cell 5: 4.5h Training Loop
    add_code("""# [Cell 5] 4.5-Hour Deep Reasoning SFT Training Loop
TOTAL_HOURS = 4.5
MAX_TRAINING_SECONDS = int(TOTAL_HOURS * 3600)  # 16,200 seconds
MAX_STEPS = 5000
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
PEAK_LR = 1.2e-4
MIN_LR = 1.0e-5
WARMUP_STEPS = 100

PROBES = [
    ("Math", "A jacket costs $150 with a 20% discount and 8% tax. What is the final price?"),
    ("Code", "Write a Python function `is_palindrome(s)` that ignores spaces and case."),
    ("Logic", "Explain step-by-step why the product of an even and an odd number is even."),
]

def generate_probe(prompt):
    model.eval()
    fmt = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
    enc = tokenizer.encode(fmt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = enc
        for _ in range(50):
            logits, _, _ = model(out)
            nxt = logits[:, -1, :] / 0.6
            for prev in set(out[0].tolist()):
                if nxt[0, prev] > 0: nxt[0, prev] /= 1.15
                else: nxt[0, prev] *= 1.15
            probs = F.softmax(nxt, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cp = torch.cumsum(sp, dim=-1)
            si_rem = cp > 0.85
            si_rem[..., 1:] = si_rem[..., :-1].clone()
            si_rem[..., 0] = 0
            rem = si_rem.scatter(1, si, si_rem)
            probs = probs.masked_fill(rem, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, tok], dim=-1)
            if tok.item() == tokenizer.eos_token_id: break
    model.train()
    return tokenizer.decode(out[0][enc.shape[1]:], skip_special_tokens=True).strip()

dataset = KaggleRandomizedReasoningDataset(tokenizer, max_seq_len=512, buffer_size=10000, max_skip=3000)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

no_decay = ["bias", "norm.weight", "scale"]
opt_params = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.1},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(opt_params, lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)

def lr_lambda(step):
    if step < WARMUP_STEPS: return float(step) / float(max(1, WARMUP_STEPS))
    prog = float(step - WARMUP_STEPS) / float(max(1, MAX_STEPS - WARMUP_STEPS))
    return max(MIN_LR / PEAK_LR, 0.5 * (1.0 + math.cos(math.pi * prog)))

scheduler = LambdaLR(optimizer, lr_lambda)
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

print("=" * 65)
print(f"🚀 Memulai Deep Reasoning SFT ({TOTAL_HOURS} Jam = {MAX_TRAINING_SECONDS}s)")
print("=" * 65)

model.train()
optimizer.zero_grad()
data_iter = iter(loader)

step = 0
tokens_trained = 0
accum_loss = 0.0
accum_aux = 0.0
best_loss = float("inf")
start_time = time.time()

history_steps = []
history_loss = []
history_aux = []

while step < MAX_STEPS:
    elapsed = time.time() - start_time
    if elapsed >= MAX_TRAINING_SECONDS:
        print(f"\\n⏰ Waktu 4.5 Jam tercapai ({elapsed:.1f}s). Menghentikan training dan menyimpan model...")
        break

    batch_loss = 0.0
    batch_aux = 0.0
    accum_toks = 0

    for _ in range(GRAD_ACCUM_STEPS):
        try: batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        in_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        logits, _, aux_loss = model(in_ids, attention_mask=attn_mask)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = lbls[..., 1:].contiguous()

        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        tot_loss = ce_loss + (aux_loss if aux_loss is not None else 0.0)

        (tot_loss / GRAD_ACCUM_STEPS).backward()
        batch_loss += ce_loss.item() / GRAD_ACCUM_STEPS
        if aux_loss is not None:
            batch_aux += aux_loss.item() / GRAD_ACCUM_STEPS

        accum_toks += (shift_labels != -100).sum().item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    step += 1
    tokens_trained += accum_toks
    accum_loss += batch_loss
    accum_aux += batch_aux

    if step % 25 == 0:
        avg_l = accum_loss / 25
        avg_a = accum_aux / 25
        c_lr = scheduler.get_last_lr()[0]
        elap = time.time() - start_time
        spd = tokens_trained / max(elap, 1e-4)
        rem_h = max(0, MAX_TRAINING_SECONDS - elap) / 3600

        history_steps.append(step)
        history_loss.append(avg_l)
        history_aux.append(avg_a)

        print(f"SFT Step {step:5d}/{MAX_STEPS} | Loss: {avg_l:.4f} (Aux: {avg_a:.4f}) | LR: {c_lr:.2e} | Speed: {spd:.0f} tok/s | Tokens: {tokens_trained/1e6:.2f}M | Left: {rem_h:.2f}h")

        if avg_l < best_loss:
            best_loss = avg_l
            torch.save({
                "step": step,
                "loss": best_loss,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, "best_reasoning_checkpoint.pt")
            print(f"  ⭐ Checkpoint Terbaik Disimpan! (Loss: {best_loss:.4f})")

        accum_loss = 0.0
        accum_aux = 0.0

    if step % 150 == 0:
        print("\\n" + "=" * 50)
        print(f"🎯 [Reasoning Probes @ Step {step}]")
        for tag, pr in PROBES:
            ans = generate_probe(pr)
            print(f"[{tag}] {ans[:140]}...")
        print("=" * 50 + "\\n")

# Save Final
torch.save({
    "step": step,
    "loss": best_loss,
    "model_state_dict": model.state_dict(),
    "config": config,
}, "nool_alpha_100m_reasoning_final.pt")
print("\\n🎉 Deep Reasoning SFT Selesai! Model disimpan di: nool_alpha_100m_reasoning_final.pt")""")

    # Cell 6: Plot Loss & MoE Balance
    add_code("""# [Cell 6] Visualisasi Loss & MoE Stability Curve
if history_steps:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_steps, history_loss, color="royalblue", lw=2, label="SFT Cross-Entropy Loss")
    plt.title("Nool-Alpha-100M Reasoning SFT Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_steps, history_aux, color="crimson", lw=2, label="HFK-MoE Router Balance Loss")
    plt.title("Router Stability (No Collapse)")
    plt.xlabel("Step")
    plt.ylabel("Aux Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("reasoning_sft_curves.png", dpi=150)
    plt.show()""")

    # Cell 7: AST Validation
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

# Load best checkpoint
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

    # Build notebook dict
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

    print(f"✅ Generated Kaggle notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_reasoning_notebook()
