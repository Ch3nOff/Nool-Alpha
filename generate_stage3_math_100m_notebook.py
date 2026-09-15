"""
Generates `nool_alpha_stage3_math_100m.ipynb` for Kaggle GPU execution.
Stage 3 Elementary Math & Reasoning Fine-Tuning Engine:
  - Resumes directly from Stage 2 checkpoint (best_stage2_checkpoint.pt / model.safetensors, Loss ~2.28)
  - 100% Mathematically Grounded Synthetic Dataset (Addition, Subtraction, Multiplication, Division)
  - Raw Equation Syntax Grounding ("1+2?", "5*5=?", "100/4=?")
  - Multi-Step Chain-of-Thought (CoT) Engine ("90+9-8-0+99=?", step-by-step intermediate scratchpad)
  - Short Conversational Greeting Anchors ("hi", "hello", "hai", "halo", "p")
  - 20% Replay Buffer of Stage 2 Factual Knowledge (History, Science, Health, Translation) to prevent catastrophic forgetting
  - Strict Prompt Loss Masking (labels = -100 on prompt)
  - Calibrated Teacher-as-a-Judge (Qwen/Qwen2.5-1.5B-Instruct) testing Math, CoT, Greetings, and Factual Retain
  - Clean Safetensors & Hugging Face Format Export
"""

import json
import os
import sys

NOTEBOOK_PATH = "nool_alpha_stage3_math_100m.ipynb"


def build_stage3_notebook():
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
    add_markdown(r"""# 🚀 Nool-Alpha-100M: Stage 3 Elementary Math & Reasoning Engine
### 👨‍🏫 Evaluator: `Qwen/Qwen2.5-1.5B-Instruct` | 🧑‍🎓 Student: `Nool-Alpha-100M` (Melanjutkan dari Stage 2 Loss ~2.28)

Notebook ini melanjutkan pelatihan dari checkpoint Stage 2 (**`best_stage2_checkpoint.pt`** atau **`model.safetensors`** dari `enx model`) yang telah menguasai persona percakapan dan pengetahuan faktual Indonesia.

---

### 🎯 Peningkatan Utama di Stage 3 (Pondasi Matematika Dasar & Penalaran Akurat):
1. **Pemberantasan Halusinasi Rumus Singkat**:
   - Model 100M pada Stage 2 mengalami halusinasi pada format ekspresi mentah seperti `1+2?` atau `5*5=?`. Stage 3 menyuntikkan puluhan ribu format persamaan matematika standar (`1+2?`, `1 + 2 = ?`, `1 + 2 = 3`, `Berapakah 1 + 2?`).
2. **Multi-Operator Chain-of-Thought (CoT) Scratchpad**:
   - Untuk perhitungan beruntun seperti `90 + 9 - 8 - 0 + 99 = ?`, model dilatih menggunakan penalaran bertahap (*step-by-step scratchpad*) agar tidak salah hitung secara autoregresif:
     ```
     Mari kita hitung langkah demi langkah:
     1. 90 + 9 = 99
     2. 99 - 8 = 91
     3. 91 - 0 = 91
     4. 91 + 99 = 190
     Jadi, hasilnya adalah 190.
     ```
3. **Penyelesaian Masalah Sapaan Singkat (*Greeting Anchors*)**:
   - Memasukkan jangkar respons untuk input pendek seperti `"hi"`, `"hello"`, `"hai"`, `"halo"`, dan `"p"` agar model tidak mengeluarkan teks prosedural acak.
4. **Anti-Catastrophic Forgetting (Replay Buffer 20%)**:
   - Mempertahankan memori faktual Stage 2 (Presiden pertama RI Soekarno, 17 Agustus 1945, cuci tangan sabun, fotosintesis, dan terjemahan) agar kepintaran umum model tidak hilang saat belajar matematika!
5. **Resume Otomatis**: Secara cerdas mendeteksi bobot Stage 2 dari `/kaggle/input` atau direktori lokal.""")

    # Cell 1: Environment Setup
    add_code("""# [Cell 1] Environment Setup & Dependencies
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!pip install -q datasets transformers accelerate bitsandbytes safetensors matplotlib rich

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
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import matplotlib.pyplot as plt

print(f"PyTorch Version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

try:
    import bitsandbytes as bnb
    print(f"✅ bitsandbytes loaded! (v{bnb.__version__})")
except ImportError:
    print("⚠️ bitsandbytes not available, standard AdamW will be used.")""")

    # Cell 2: Nool-Alpha-100M Architecture
    add_code("""# [Cell 2] Nool-Alpha-100M Architecture (GSLA + HFK-MoE)
from dataclasses import dataclass
from typing import Optional, Tuple, List

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
    gradient_checkpointing: bool = False

    @classmethod
    def nool_100m(cls, vocab_size: int = 50257):
        return cls(vocab_size=vocab_size)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class DecoupledRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 500000.0):
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
        return self.cos_cached[:seq_len, :].to(dtype=x.dtype), self.sin_cached[:seq_len, :].to(dtype=x.dtype)

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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
        self.is_swa = (layer_idx % config.swa_interval != (config.swa_interval - 1))
        self.window_size = config.sliding_window if self.is_swa else None

        self.w_dk = nn.Linear(self.d_model, self.d_c, bias=False)
        self.w_pos = nn.Linear(self.d_model, self.d_pe, bias=False)
        self.w_q = nn.Linear(self.d_model, self.num_heads * (self.head_dim + self.d_pe), bias=False)
        self.w_uk = nn.Linear(self.d_c, self.num_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(self.d_c, self.num_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.rope = DecoupledRotaryEmbedding(dim=self.d_pe, max_seq_len=config.max_position_embeddings, theta=config.rope_theta)

    def forward(self, x, attention_mask=None, kv_cache=None, use_cache=False):
        B, S, _ = x.shape
        c_kv = self.w_dk(x)
        k_pos = self.w_pos(x)
        cos, sin = self.rope(k_pos, S)
        k_pos = self.rope.apply_rope(k_pos, cos, sin)

        if kv_cache is not None:
            prev_c_kv, prev_k_pos = kv_cache
            c_kv = torch.cat([prev_c_kv, c_kv], dim=1)
            k_pos = torch.cat([prev_k_pos, k_pos], dim=1)
        current_cache = (c_kv, k_pos) if use_cache else None

        S_kv = c_kv.shape[1]
        K_content = self.w_uk(c_kv).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_uv(c_kv).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        K_pos_expanded = k_pos.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q_all = self.w_q(x).view(B, S, self.num_heads, self.head_dim + self.d_pe).transpose(1, 2)
        Q_c = q_all[..., :self.head_dim]
        Q_pos = self.rope.apply_rope(q_all[..., self.head_dim:], cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim + self.d_pe)
        scores_content = torch.matmul(Q_c, K_content.transpose(-1, -2))
        scores_pos = torch.matmul(Q_pos, K_pos_expanded.transpose(-1, -2))
        scores = (scores_content + scores_pos) * scale

        causal_mask = torch.triu(torch.full((S, S_kv), float("-inf"), device=x.device), diagonal=S_kv - S + 1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)

        if self.is_swa and self.window_size is not None and S_kv > self.window_size:
            row_idx = torch.arange(S, device=x.device).unsqueeze(1)
            col_idx = torch.arange(S_kv, device=x.device).unsqueeze(0)
            swa_mask = (col_idx < (row_idx + S_kv - S - self.window_size))
            scores = scores.masked_fill(swa_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, S, -1)
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
        self.config = config
        self.d_model = config.d_model
        self.shared_ffn_dim = config.shared_ffn_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.expert_rank = config.expert_rank
        self.aux_coeff = config.moe_aux_loss_coeff

        self.shared_gate = nn.Linear(self.d_model, self.shared_ffn_dim, bias=False)
        self.shared_up = nn.Linear(self.d_model, self.shared_ffn_dim, bias=False)
        self.shared_down = nn.Linear(self.shared_ffn_dim, self.d_model, bias=False)

        self.experts = nn.ModuleList([FactorizedExpert(self.d_model, self.expert_rank) for _ in range(self.num_experts)])
        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.shared_down(F.silu(self.shared_gate(x)) * self.shared_up(x))
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        dynamic_out = torch.zeros_like(x)
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        indices_flat = indices.view(-1, self.top_k)
        weights_flat = weights.view(-1, self.top_k)

        for k in range(self.top_k):
            exp_indices = indices_flat[:, k]
            exp_weights = weights_flat[:, k].unsqueeze(-1)
            for e_idx in range(self.num_experts):
                mask = (exp_indices == e_idx)
                if mask.any():
                    dynamic_out.view(-1, D)[mask] += exp_weights[mask] * self.experts[e_idx](x_flat[mask])

        P = router_probs.view(-1, self.num_experts).mean(dim=0)
        mask_flat = F.one_hot(indices_flat, num_classes=self.num_experts).float().sum(dim=1)
        f = mask_flat.mean(dim=0) / self.top_k
        aux_loss = self.aux_coeff * self.num_experts * torch.sum(P * f)

        return shared_out + dynamic_out, aux_loss

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
        m_out, aux = self.moe(self.norm2(x))
        x = x + m_out
        return x, aux, cache

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
            if use_cache: next_caches.append(c)

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

print("[OK] Nool-Alpha-100M Architecture Defined!")""")

    # Cell 3: Dual Model Initialization (Teacher 4-bit + Student FP32 Master)
    add_code("""# [Cell 3] Dual Model Initialization: Teacher (4-bit) & Student (FP32 Master with AMP)
TEACHER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"👨‍🏫 Memuat Model Evaluator: '{TEACHER_MODEL_ID}' dalam 4-bit NF4...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True
)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

t_vram = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
print(f"✅ Guru Evaluator Qwen-1.5B Berhasil Dimuat! VRAM: {t_vram:.2f} GB")

print("\\n🧑‍🎓 Inisialisasi Model Murid: 'Nool-Alpha-100M'...")
student_config = NoolAlphaConfig.nool_100m(vocab_size=50257)
student_model = NoolAlphaForCausalLM(student_config).to(device=device)

student_tokenizer = AutoTokenizer.from_pretrained("gpt2")
if student_tokenizer.pad_token_id is None:
    student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

tot_p, act_p = student_model.get_num_params()
s_vram = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
print("=" * 65)
print(f"📊 DUAL MODEL VRAM ALLOCATION:")
print(f"  • Evaluator (Qwen-2.5-1.5B 4-bit) : ~{t_vram:.2f} GB")
print(f"  • Murid (Nool-Alpha-100M FP32)    : ~{s_vram - t_vram:.2f} GB ({tot_p/1e6:.1f}M total, {act_p/1e6:.1f}M active)")
print(f"  • Total VRAM Terpakai             : {s_vram:.2f} GB / 14.56 GB")
print(f"  • Sisa Headroom Bebas             : ~{14.56 - s_vram:.2f} GB (Sangat Aman! 🟢)")
print("=" * 65)""")

    # Cell 4: Checkpoint Auto-Discovery & Resume
    add_code("""# [Cell 4] Checkpoint Auto-Discovery: Memuat Bobot Stage 2 (best_stage2_checkpoint.pt / model.safetensors)
def find_student_checkpoint():
    search_dirs = [
        "/kaggle/input",
        "/kaggle/working",
        "."
    ]
    priority_names = [
        "best_stage2_checkpoint.pt",
        "nool_alpha_100m_stage2_final.pt",
        "model.safetensors",
        "best_teacher_distill_checkpoint.pt",
        "best_checkpoint.pt"
    ]
    for p_name in priority_names:
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                for root, _, files in os.walk(s_dir):
                    if p_name in files:
                        return os.path.join(root, p_name)
    return None

CKPT_PATH = find_student_checkpoint()
base_loss = float("inf")
base_step = 0

if CKPT_PATH and os.path.exists(CKPT_PATH):
    print(f"🔄 Checkpoint Stage 2 Ditemukan: '{CKPT_PATH}'")
    if CKPT_PATH.endswith(".safetensors"):
        from safetensors.torch import load_file
        raw_state = load_file(CKPT_PATH)
    else:
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        raw_state = ckpt.get("model_state_dict", ckpt)
        base_loss = ckpt.get("loss", float("inf"))
        base_step = ckpt.get("step", 0)
    
    m_dict = student_model.state_dict()
    matched = {k: v.to(device=device, dtype=torch.float32) for k, v in raw_state.items() if k in m_dict and v.shape == m_dict[k].shape}
    m_dict.update(matched)
    student_model.load_state_dict(m_dict)
    print(f"✅ Berhasil memuat {len(matched)} matching tensors!")
    print(f"📈 Melanjutkan dari Baseline: Step {base_step} | Baseline Loss: {base_loss:.4f}")
else:
    print("ℹ️ Checkpoint tidak ditemukan di /kaggle/input. Memulai dari inisialisasi awal.")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()""")

    # Cell 5: Stage 3 Multi-Stream Math & Reasoning Dataset Engine
    add_code("""# [Cell 5] Stage 3 Multi-Stream Math & Reasoning Dataset Engine

# 1. Jangkar Sapaan & Persona (Memperbaiki respons "hi" yang sebelumnya halu ke teks prosedural)
GREETING_ANCHORS = [
    ("hi", "Halo! Senang bertemu denganmu. Ada yang bisa saya bantu hari ini?"),
    ("hello", "Halo! Saya Nool-Alpha, asisten kecerdasan buatan. Ada yang bisa saya bantu?"),
    ("hai", "Hai! Apa kabar? Ada yang ingin Anda diskusikan atau tanyakan?"),
    ("halo", "Halo! Saya Nool-Alpha. Senang bisa membantu Anda."),
    ("p", "Halo! Ada yang bisa saya bantu?"),
    ("selamat pagi", "Selamat pagi! Semoga hari Anda menyenangkan dan penuh semangat. Ada yang bisa saya bantu?"),
    ("selamat siang", "Selamat siang! Bagaimana harimu? Ada yang bisa saya bantu?"),
    ("selamat malam", "Selamat malam! Ada yang bisa saya bantu sebelum Anda beristirahat?"),
    ("halo apa kabar?", "Halo! Kabar saya sangat baik. Saya siap membantu Anda menjawab pertanyaan atau berdiskusi."),
    ("siapa kamu?", "Saya adalah Nool-Alpha, model kecerdasan buatan yang dikembangkan untuk membantu menjawab pertanyaan, bernalar, dan belajar secara interaktif."),
    ("siapa namamu?", "Nama saya Nool-Alpha, asisten kecerdasan buatan Anda."),
    ("apa yang bisa kamu lakukan?", "Saya dapat membantu Anda menjawab pertanyaan umum, menyelesaikan perhitungan matematika, menjelaskan konsep sains, menerjemahkan bahasa, dan berdiskusi ramah.")
]

# 2. Buffer Replay Stage 2 (Anti-Catastrophic Forgetting)
STAGE2_FACTUAL_REPLAY = [
    ("Siapa presiden pertama Republik Indonesia?", "Presiden pertama Republik Indonesia adalah Ir. Soekarno, dengan wakil presiden Drs. Mohammad Hatta."),
    ("Kapan Indonesia merdeka?", "Indonesia memproklamasikan kemerdekaannya pada tanggal 17 Agustus 1945."),
    ("Apa ibukota negara Indonesia?", "Ibukota negara Indonesia saat ini adalah Jakarta, dan sedang dalam proses transisi ke Ibu Kota Nusantara (IKN) di Kalimantan Timur."),
    ("Apa dasar negara Indonesia?", "Dasar negara Indonesia adalah Pancasila, yang terdiri dari lima sila sebagai panduan kehidupan berbangsa dan bernegara."),
    ("Mengapa kita harus mencuci tangan dengan sabun sebelum makan?", "Mencuci tangan dengan sabun sebelum makan sangat penting untuk membunuh kuman, bakteri, dan virus yang menempel di tangan, sehingga mencegah penyakit saluran pencernaan."),
    ("Bagaimana tumbuhan menghasilkan makanan sendiri?", "Tumbuhan menghasilkan makanan sendiri melalui proses fotosintesis, di mana daun menyerap sinar matahari, air dari akar, dan karbon dioksida dari udara untuk menghasilkan glukosa dan oksigen."),
    ("Apa fungsi jantung pada tubuh manusia?", "Fungsi utama jantung adalah memompa darah yang kaya akan oksigen dan nutrisi ke seluruh organ tubuh, lalu memompa darah kotor kembali ke paru-paru untuk dibersihkan."),
    ("Jelaskan mengapa air laut rasanya asin.", "Air laut terasa asin karena mengandung mineral terlarut, terutama natrium klorida (garam dapur), yang terbawa oleh aliran sungai dari batuan daratan selama jutaan tahun."),
    ("Mengapa langit berwarna biru pada siang hari?", "Langit berwarna biru karena fenomena Hamburan Rayleigh (Rayleigh Scattering), di mana molekul-molekul di atmosfer menghamburkan cahaya matahari dengan panjang gelombang pendek (biru) lebih banyak daripada cahaya merah."),
    ("Terjemahkan ke bahasa Inggris: Selamat pagi, semoga harimu menyenangkan!", "Good morning, have a wonderful day!"),
    ("Terjemahkan ke bahasa Inggris: Terima kasih banyak atas bantuanmu.", "Thank you very much for your help."),
    ("Terjemahkan ke bahasa Indonesia: Good afternoon, how can I assist you today?", "Selamat siang, bagaimana saya bisa membantu Anda hari ini?"),
    ("Tuliskan fungsi Python sederhana untuk menjumlahkan dua angka.", "```python\\ndef tambah(a, b):\\n    return a + b\\n```"),
    ("Sebutkan 3 tips menjaga kesehatan tubuh.", "1. Konsumsi makanan bergizi seimbang.\\n2. Rutin berolahraga minimal 30 menit sehari.\\n3. Istirahat dan tidur yang cukup (7-8 jam setiap malam).")
]

# 3. Dynamic Elementary Math Generator (100% Mathematically Correct)
def generate_elementary_math(rng: random.Random) -> Tuple[str, str]:
    op = rng.choice(["+", "-", "*", "/"])
    
    if op == "+":
        tier = rng.random()
        if tier < 0.35:
            a = rng.randint(0, 9)
            b = rng.randint(0, 9)
        elif tier < 0.75:
            a = rng.randint(10, 99)
            b = rng.randint(1, 99)
        else:
            a = rng.randint(100, 999)
            b = rng.randint(10, 500)
        ans = a + b
        
        templates = [
            (f"{a}+{b}?", f"{a} + {b} = {ans}"),
            (f"{a} + {b}?", f"{a} + {b} = {ans}"),
            (f"{a}+{b} = ?", f"{a} + {b} = {ans}"),
            (f"{a} + {b} = ?", f"{a} + {b} = {ans}"),
            (f"{a}+{b}=", f"{a} + {b} = {ans}"),
            (f"{a} + {b} =", f"{a} + {b} = {ans}"),
            (f"Berapakah {a} + {b}?", f"{a} + {b} = {ans}"),
            (f"Berapa {a} + {b}?", f"{a} + {b} = {ans}"),
            (f"Berapakah hasil dari {a} ditambah {b}?", f"Hasil dari {a} ditambah {b} adalah {ans} ({a} + {b} = {ans})."),
            (f"Berapa {a} ditambah {b}?", f"{a} ditambah {b} adalah {ans}."),
            (f"Hitunglah {a} + {b}", f"{a} + {b} = {ans}"),
            (f"Hasil penjumlahan {a} dan {b} adalah?", f"Hasil penjumlahan {a} dan {b} adalah {ans} ({a} + {b} = {ans}).")
        ]
        return rng.choice(templates)
        
    elif op == "-":
        tier = rng.random()
        if tier < 0.35:
            a = rng.randint(0, 9)
            b = rng.randint(0, a)
        elif tier < 0.75:
            a = rng.randint(10, 99)
            b = rng.randint(1, a)
        else:
            a = rng.randint(100, 999)
            b = rng.randint(10, a)
        ans = a - b
        
        templates = [
            (f"{a}-{b}?", f"{a} - {b} = {ans}"),
            (f"{a} - {b}?", f"{a} - {b} = {ans}"),
            (f"{a}-{b} = ?", f"{a} - {b} = {ans}"),
            (f"{a} - {b} = ?", f"{a} - {b} = {ans}"),
            (f"{a}-{b}=", f"{a} - {b} = {ans}"),
            (f"{a} - {b} =", f"{a} - {b} = {ans}"),
            (f"Berapakah {a} - {b}?", f"{a} - {b} = {ans}"),
            (f"Berapa {a} - {b}?", f"{a} - {b} = {ans}"),
            (f"Berapakah hasil dari {a} dikurang {b}?", f"Hasil dari {a} dikurang {b} adalah {ans} ({a} - {b} = {ans})."),
            (f"Berapa {a} dikurangi {b}?", f"{a} dikurangi {b} adalah {ans}."),
            (f"Hitunglah {a} - {b}", f"{a} - {b} = {ans}"),
            (f"Berapakah selisih antara {a} dan {b}?", f"Selisih antara {a} dan {b} adalah {ans} ({a} - {b} = {ans}).")
        ]
        return rng.choice(templates)
        
    elif op == "*":
        tier = rng.random()
        if tier < 0.60:
            a = rng.randint(0, 10)
            b = rng.randint(0, 10)
        elif tier < 0.85:
            a = rng.randint(11, 25)
            b = rng.randint(1, 10)
        else:
            a = rng.choice([10, 20, 25, 50, 100])
            b = rng.choice([2, 3, 4, 5, 10])
        ans = a * b
        
        templates = [
            (f"{a}*{b}?", f"{a} * {b} = {ans}"),
            (f"{a} * {b}?", f"{a} * {b} = {ans}"),
            (f"{a}x{b}?", f"{a} x {b} = {ans}"),
            (f"{a} x {b} = ?", f"{a} x {b} = {ans}"),
            (f"{a} * {b} = ?", f"{a} * {b} = {ans}"),
            (f"{a} * {b} =", f"{a} * {b} = {ans}"),
            (f"Berapakah {a} x {b}?", f"{a} x {b} = {ans}"),
            (f"Berapa {a} * {b}?", f"{a} * {b} = {ans}"),
            (f"Berapakah hasil dari {a} dikali {b}?", f"Hasil dari {a} dikali {b} adalah {ans} ({a} x {b} = {ans})."),
            (f"Berapa {a} dikalikan {b}?", f"{a} dikalikan {b} adalah {ans}."),
            (f"Hitung perkalian {a} x {b}", f"{a} x {b} = {ans}")
        ]
        return rng.choice(templates)
        
    else:  # Division
        b = rng.randint(1, 12)
        q = rng.randint(1, 20)
        a = b * q
        ans = q
        
        templates = [
            (f"{a}/{b}?", f"{a} / {b} = {ans}"),
            (f"{a} / {b}?", f"{a} / {b} = {ans}"),
            (f"{a}:{b}?", f"{a} : {b} = {ans}"),
            (f"{a} / {b} = ?", f"{a} / {b} = {ans}"),
            (f"{a} : {b} = ?", f"{a} : {b} = {ans}"),
            (f"{a} / {b} =", f"{a} / {b} = {ans}"),
            (f"Berapakah {a} dibagi {b}?", f"Hasil dari {a} dibagi {b} adalah {ans} ({a} / {b} = {ans})."),
            (f"Berapa {a} : {b}?", f"{a} : {b} = {ans}"),
            (f"Hitung pembagian {a} / {b}", f"{a} / {b} = {ans}")
        ]
        return rng.choice(templates)

# 4. Multi-Operator Chain-of-Thought (CoT) Generator (Intermediate Scratchpads)
def generate_cot_math(rng: random.Random) -> Tuple[str, str]:
    num_ops = rng.randint(2, 4)
    start = rng.choice([rng.randint(1, 99), rng.randint(1, 25)])
    current = start
    
    tokens = [str(start)]
    steps = []
    
    for i in range(num_ops):
        op = rng.choice(["+", "-", "+"])
        if op == "+":
            val = rng.randint(0, 99) if current < 100 else rng.randint(0, 50)
            next_val = current + val
            steps.append(f"{i+1}. {current} + {val} = {next_val}")
            tokens.extend(["+", str(val)])
            current = next_val
        else:
            val = rng.randint(0, min(current, 50))
            next_val = current - val
            steps.append(f"{i+1}. {current} - {val} = {next_val}")
            tokens.extend(["-", str(val)])
            current = next_val
            
    expr = " ".join(tokens)
    steps_txt = "\\n".join(steps)
    
    prompt_templates = [
        f"Hitunglah: {expr} = ?",
        f"{expr} = ?",
        f"{expr}?",
        f"Berapakah hasil dari {expr}?",
        f"Selesaikan perhitungan berikut secara bertahap: {expr}"
    ]
    prompt = rng.choice(prompt_templates)
    
    resp = f"Mari kita hitung langkah demi langkah:\\n{steps_txt}\\nJadi, hasilnya adalah {current}."
    return prompt, resp

class Stage3MathDataset(IterableDataset):
    \"\"\"
    Multi-stream Stage 3 Dataset:
      1. 40% Elementary Math & Equation Grounding (1+2?, a*b, division)
      2. 25% Multi-Step Chain-of-Thought (CoT) Math (90+9-8-0+99=?)
      3. 15% Short Greeting & Conversational Anchors ("hi", "hello", "siapa kamu")
      4. 10% Factual Replay Buffer (Soekarno, Sains, Kesehatan)
      5. 10% Alpaca Indonesian Streaming (Menjaga kelancaran tata bahasa)
      
    STRICT PROMPT LOSS MASKING:
      labels = [-100] * len(prompt_ids) + resp_ids
    \"\"\"
    def __init__(self, tokenizer, max_seq_len: int = 512, reservoir_size: int = 64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reservoir_size = reservoir_size
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.rng = random.Random(42)

    def _stream_elementary_math(self):
        while True:
            yield generate_elementary_math(self.rng)

    def _stream_cot_math(self):
        while True:
            yield generate_cot_math(self.rng)

    def _stream_greetings(self):
        while True:
            shuffled = list(GREETING_ANCHORS)
            self.rng.shuffle(shuffled)
            for p, r in shuffled:
                yield p, r

    def _stream_factual_replay(self):
        while True:
            shuffled = list(STAGE2_FACTUAL_REPLAY)
            self.rng.shuffle(shuffled)
            for p, r in shuffled:
                yield p, r

    def _stream_alpaca(self):
        while True:
            try:
                ds = load_dataset("FreedomIntelligence/alpaca-gpt4-indonesian", split="train", streaming=True)
                for item in ds:
                    convs = item.get("conversations", [])
                    p, r = "", ""
                    if convs:
                        for c in convs:
                            sender = c.get("from", "").lower(); val = c.get("value", "").strip()
                            if sender in ["user", "human"] and not p: p = val
                            elif sender in ["assistant", "gpt"] and p and not r: r = val; break
                    else:
                        inst = item.get("instruction", "").strip(); inp = item.get("input", "").strip(); out = item.get("output", "").strip()
                        if inst and out: p = f"{inst}\\nKonteks: {inp}" if inp else inst; r = out
                    if p and r and len(r) > 10:
                        yield p[:1500].strip(), r[:2000].strip()
            except Exception:
                continue

    def _tokenize(self, prompt: str, resp: str):
        prompt_txt = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
        prompt_ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
        resp_ids = self.tokenizer.encode(resp, add_special_tokens=False) + [self.eos_token_id]

        total = len(prompt_ids) + len(resp_ids)
        if total > self.max_seq_len:
            max_resp = self.max_seq_len - len(prompt_ids)
            if max_resp < 24: return None
            resp_ids = resp_ids[:max_resp - 1] + [self.eos_token_id]

        input_ids = prompt_ids + resp_ids
        # STRICT MASKING: HANYA token jawaban yang dihitung gradiennya!
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
            ("elementary_math", self._stream_elementary_math()),
            ("cot_math", self._stream_cot_math()),
            ("greetings", self._stream_greetings()),
            ("factual_replay", self._stream_factual_replay()),
            ("alpaca_id", self._stream_alpaca()),
        ]
        weights = [0.40, 0.25, 0.15, 0.10, 0.10]
        stream_indices = list(range(len(streams)))

        def get_sample():
            while True:
                idx = self.rng.choices(stream_indices, weights=weights, k=1)[0]
                _, st = streams[idx]
                try:
                    p, r = next(st)
                    tok = self._tokenize(p, r)
                    if tok is not None: return tok
                except Exception:
                    continue

        reservoir = [get_sample() for _ in range(self.reservoir_size)]
        while True:
            pick = self.rng.randint(0, len(reservoir) - 1)
            sample = reservoir[pick]
            reservoir[pick] = get_sample()
            yield sample

print("✅ Stage 3 Math & Reasoning Dataset Engine Siap!")""")

    # Cell 6: Training Loop with Strict Teacher-as-a-Judge
    add_code("""# [Cell 6] Training Loop with Strict Teacher-as-a-Judge Evaluation
OUTPUT_DIR = "/kaggle/working/nool_alpha_100m_stage3_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameter Stage 3 Fine-Tuning
MAX_TRAINING_HOURS = 3.5
TARGET_MAX_STEPS = 3500
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8    # Effective batch = 16 seqs (8,192 tok/step)
PEAK_LR = 1.0e-4        # LR fokus stabilisasi bobot matematika
MIN_LR = 1.0e-5
WARMUP_STEPS = 60
LOG_INTERVAL = 25
EVAL_INTERVAL = 150

dataset = Stage3MathDataset(student_tokenizer, max_seq_len=512, reservoir_size=64)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
data_iter = iter(dataloader)

decay_params = [p for n, p in student_model.named_parameters() if p.requires_grad and p.dim() >= 2]
nodecay_params = [p for n, p in student_model.named_parameters() if p.requires_grad and p.dim() < 2]

try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit([
        {"params": decay_params, "weight_decay": 0.05},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)
    print("🚀 Menggunakan 8-bit AdamW!")
except Exception:
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.05},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)

scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

def scheduler_fn(step_idx: int):
    if step_idx < WARMUP_STEPS:
        return float(step_idx) / float(max(1, WARMUP_STEPS))
    prog = float(step_idx - WARMUP_STEPS) / float(max(1, TARGET_MAX_STEPS - WARMUP_STEPS))
    return MIN_LR / PEAK_LR + 0.5 * (1.0 - MIN_LR / PEAK_LR) * (1.0 + math.cos(math.pi * prog))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)

EVAL_PROBES = [
    ("Math Persamaan Singkat", "1+2?"),
    ("Math Multi-Step CoT", "90 + 9 - 8 - 0 + 99 = ?"),
    ("Sapaan Singkat (Anchor)", "hi"),
    ("Penjumlahan Standar", "Berapakah hasil dari 25 ditambah 75?"),
    ("Sejarah RI (Replay Memori)", "Siapa presiden pertama Republik Indonesia?"),
    ("Sains & Kesehatan (Replay Memori)", "Mengapa kita harus mencuci tangan dengan sabun sebelum makan?"),
]

def student_generate(prompt: str, max_tokens: int = 70):
    student_model.eval()
    fmt = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
    inp = student_tokenizer(fmt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = inp.shape[1]

    with torch.no_grad():
        for _ in range(max_tokens):
            idx = inp[:, -512:] if inp.shape[1] > 512 else inp
            logits, _, _, _ = student_model(idx)
            next_logits = logits[:, -1, :] / 0.3  # Suhu rendah untuk presisi logika

            # Repetition penalty (1.2)
            for tid in set(inp[0].tolist()):
                if next_logits[0, tid] > 0: next_logits[0, tid] /= 1.2
                else: next_logits[0, tid] *= 1.2

            # Top-P sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs > 0.85
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = 0
            indices_to_remove = remove_mask.scatter(1, sorted_indices, remove_mask)
            next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            inp = torch.cat([inp, next_token], dim=1)
            if next_token.item() == student_tokenizer.eos_token_id:
                break

    decoded = student_tokenizer.decode(inp[0].tolist(), skip_special_tokens=False)
    student_model.train()
    if "### Response:\\n" in decoded:
        return decoded.split("### Response:\\n", 1)[1].replace("<|endoftext|>", "").strip()
    return decoded[prompt_len:].replace("<|endoftext|>", "").strip()

def teacher_strict_judge(prompt: str, student_answer: str) -> Tuple[int, str]:
    \"\"\"Evaluator Qwen-1.5B dengan kriteria ketat untuk matematika dan fakta.\"\"\"
    judge_prompt = f\"\"\"Kamu adalah evaluator kritis yang menilai kebenaran jawaban model bahasa.
    
Pertanyaan: {prompt}
Jawaban Murid: {student_answer}

KRITERIA PENILAIAN:
- Beri 1/5 jika jawaban salah hitung, melantur, halu matematika, atau tidak nyambung.
- Beri 2/5 jika jawaban mencoba menjawab tetapi perhitungan salah atau faktanya keliru.
- Beri 3/5 jika penalaran hampir benar tetapi hasil akhir sedikit meleset.
- Beri 4/5 jika jawaban benar, perhitungan akurat, atau sapaan tepat.
- Beri 5/5 jika jawaban sempurna, runtut langkahnya, dan memuaskan.

Format Wajib:
Rating: X/5
Kritik: (satu kalimat penilaian jujur)\"\"\"

    msgs = [{"role": "user", "content": judge_prompt}]
    text = teacher_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    t_inp = teacher_tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        t_out = teacher_model.generate(**t_inp, max_new_tokens=65, temperature=0.1)
        resp = teacher_tokenizer.decode(t_out[0][t_inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

    stars = 1
    match = re.search(r"Rating:\\s*(\\d)/5", resp) or re.search(r"(\\d)/5", resp)
    if match:
        stars = max(1, min(5, int(match.group(1))))
    return stars, resp

print("=" * 70)
print(f"🚀 Memulai Stage 3 Math & Reasoning Pelatihan ({MAX_TRAINING_HOURS}h Budget = {int(MAX_TRAINING_HOURS*3600)}s)")
print(f"Batch Efektif: {BATCH_SIZE * GRAD_ACCUM_STEPS} seqs ({BATCH_SIZE * GRAD_ACCUM_STEPS * 512} tok/step)")
print(f"Starting Loss Baseline: {base_loss:.4f}")
print("=" * 70)

start_time = time.time()
max_seconds = MAX_TRAINING_HOURS * 3600.0
step = 0
running_loss = 0.0
running_aux = 0.0
best_loss = base_loss if base_loss < float("inf") else float("inf")
session_tokens = 0
t_prev = time.time()

log_steps = []
log_losses = []
log_aux = []

student_model.train()
while step < TARGET_MAX_STEPS:
    elapsed = time.time() - start_time
    if elapsed >= max_seconds:
        print(f"\\n⏱️ Batas waktu latihan tercapai ({MAX_TRAINING_HOURS} jam).")
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

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, loss, aux_loss, _ = student_model(input_ids, labels)

        scaled_loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(scaled_loss).backward()

        step_loss += loss.item() / GRAD_ACCUM_STEPS
        step_aux += aux_loss.item() / GRAD_ACCUM_STEPS
        session_tokens += input_ids.numel()

        del logits, loss, aux_loss

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
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

        print(f"Stage 3 Step {step:4d}/{TARGET_MAX_STEPS} | Loss: {avg_l:.4f} (Aux: {avg_a:.4f}) | LR: {cur_lr:.2e} | Speed: {speed:.0f} tok/s | Tokens: {session_tokens/1e6:.2f}M | Sisa: {rem_h:.2f}h")

        if avg_l < best_loss:
            best_loss = avg_l
            torch.save({
                "step": base_step + step,
                "loss": best_loss,
                "model_state_dict": student_model.state_dict(),
                "config": student_config,
            }, os.path.join(OUTPUT_DIR, "best_stage3_math_checkpoint.pt"))
            print(f"  ⭐ Checkpoint Terbaik Baru Disimpan! (Loss: {best_loss:.4f})")

    # Evaluator Teacher Evaluation
    if step % EVAL_INTERVAL == 0:
        print("\\n" + "=" * 65)
        print(f"👨‍🏫 [PENILAIAN EVALUATOR @ Step {step}]")
        print("=" * 65)
        for cat, pr in EVAL_PROBES:
            ans = student_generate(pr, max_tokens=65)
            stars, critique = teacher_strict_judge(pr, ans)
            stars_str = "⭐" * stars + "☆" * (5 - stars)
            print(f"📌 [{cat}]")
            print(f"  Q: {pr}")
            print(f"  A (Nool-Alpha): {ans}")
            print(f"  👨‍🏫 Nilai Evaluator: {stars_str} ({stars}/5)")
            print(f"  💬 Catatan: {critique.replace(chr(10), ' ')[:120]}...")
            print("-" * 65)
        print("=" * 65 + "\\n")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Final Checkpoint
final_path = os.path.join(OUTPUT_DIR, "nool_alpha_100m_stage3_math_final.pt")
torch.save({
    "step": base_step + step,
    "loss": best_loss,
    "model_state_dict": student_model.state_dict(),
    "config": student_config,
}, final_path)
print(f"\\n🎉 Stage 3 Selesai! Model disimpan di: '{final_path}'")""")

    # Cell 7: Plots
    add_code("""# [Cell 7] Visualisasi Kurva Loss & Stabilitas MoE Stage 3
if log_steps:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(log_steps, log_losses, color="forestgreen", lw=2, label="Stage 3 Math Loss")
    if base_loss < float("inf"):
        plt.axhline(y=base_loss, color="gray", linestyle="--", alpha=0.7, label=f"Stage 2 Baseline ({base_loss:.2f})")
    plt.title("Nool-Alpha-100M Stage 3 Math Convergence")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(log_steps, log_aux, color="darkorange", lw=2, label="Router Aux Loss (8 Experts)")
    plt.title("MoE Router Balance Stability")
    plt.xlabel("Step")
    plt.ylabel("Aux Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("stage3_math_curves.png", dpi=150)
    plt.show()""")

    # Cell 8: Interactive Test Probe
    add_code("""# [Cell 8] Uji Mandiri: Pengujian Presisi Matematika & Fakta Nool-Alpha-100M Stage 3
print("=" * 65)
print("💬 Sesi Uji Coba Presisi Matematika & Persona (Nool-Alpha-100M Stage 3)")
print("=" * 65)

test_questions = [
    "1+2?",
    "90 + 9 - 8 - 0 + 99 = ?",
    "hi",
    "Berapakah 7 dikali 8?",
    "Berapakah hasil dari 100 dikurang 35?",
    "Siapa presiden pertama Republik Indonesia?",
    "Mengapa kita harus mencuci tangan dengan sabun sebelum makan?"
]

for q in test_questions:
    res = student_generate(q, max_tokens=75)
    print(f"👤 User    : {q}")
    print(f"🤖 Nool-100M: {res}")
    print("-" * 65)""")

    # Cell 9: Safetensors Export
    add_code("""# [Cell 9] Ekspor Final ke Format Safetensors & Hugging Face
from safetensors.torch import save_file

export_dir = "exported_nool_alpha_100m_stage3_math"
os.makedirs(export_dir, exist_ok=True)

best_ckpt_file = os.path.join(OUTPUT_DIR, "best_stage3_math_checkpoint.pt")
if not os.path.exists(best_ckpt_file):
    best_ckpt_file = final_path

ckpt = torch.load(best_ckpt_file, map_location="cpu", weights_only=False)
raw_state = ckpt.get("model_state_dict", ckpt)

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
    "stage": "stage3_math_reasoning",
    "evaluator_model": TEACHER_MODEL_ID,
    "best_loss": ckpt.get("loss", "N/A"),
    "total_steps": ckpt.get("step", "N/A")
}

with open(os.path.join(export_dir, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=2)

student_tokenizer.save_pretrained(export_dir)
print(f"✅ Model Stage 3 tersimpan dalam format Safetensors di: '{export_dir}/'")
print(f"File model: {os.path.join(export_dir, 'model.safetensors')} ({os.path.getsize(os.path.join(export_dir, 'model.safetensors')) / (1024*1024):.2f} MB)")
print("🚀 Siap diunduh dan dijalankan di Local Web UI!")""")

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

    print(f"[OK] Generated Stage 3 Math & Reasoning Notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_stage3_notebook()
