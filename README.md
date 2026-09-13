# Nool-Alpha: Architectural Implementation & Empirical Benchmarks

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Nool--Alpha--100M--Chat-blue)](https://huggingface.co/CH3NDev/Nool-Alpha-100M-Chat)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Empirical Analysis](https://img.shields.io/badge/Empirical%20Report-Verified-success)](NOOL_ALPHA_100M_EMPIRICAL_ANALYSIS.md)

**Official Hugging Face Model:** [huggingface.co/CH3NDev/Nool-Alpha-100M-Chat](https://huggingface.co/CH3NDev/Nool-Alpha-100M-Chat)  
**Empirical Benchmark Report:** [`NOOL_ALPHA_100M_EMPIRICAL_ANALYSIS.md`](NOOL_ALPHA_100M_EMPIRICAL_ANALYSIS.md)

Implementation of **Nool-Alpha** featuring **Grouped-Subspace Latent Attention (GSLA)** with **Decoupled RoPE** and **Heterogeneous Factorized MoE (HFK-MoE)**, reducing KV-cache VRAM consumption by **87.8%** and active parameter FLOPs by **21.3%** compared to standard dense 100M–125M architectures (e.g. GPT-2 Small 124M).

---

## 🏛️ Architecture Blueprint Overview

Nool-Alpha-1.5B is engineered to address two major bottlenecks in standard small-to-medium language models: **KV-cache memory bloat** and **FFN capacity fragmentation**.

```
Input Tokens ──> Embedding (tied) ──> [ Transformer Layer × 24 ] ──> Final RMSNorm ──(+)──> Logit Soft-Capping ──> Logits
                     │                                                                  ▲
                     └── RMSNorm(x_0) ───[ Global Residual Highway: tanh(α) ]───────────┘
```

### 1. Grouped-Subspace Latent Attention (GSLA) with Decoupled RoPE
- **Content Latent Compression ($d_c = 448$)**: Compresses semantic token representation into a latent vector $c_t = x_t W_{DK}$, where $W_{DK} \in \mathbb{R}^{2048 \times 448}$.
- **Decoupled RoPE Key Subspace ($d_{pe} = 64$)**: Dedicated position subspace $K_t^{pe} = R_{\Theta, t}(x_t W_{pos})$ with base frequency $\Theta = 500,000$.
- **KV Cache Allocation**: Only $[c_t \parallel K_t^{pe}] \in \mathbb{R}^{512}$ is allocated to cache per token (equivalent to ~2 standard GQA heads).
- **Query Absorption**: The Key up-projection matrix $W_{UK, h}$ is absorbed directly into the Query:
  $$\tilde{Q}_{t, h} = Q_{t, h}^{val} W_{UK, h}^T \in \mathbb{R}^{448}$$
  Attention scores compute directly on the 448-dim latent cache without decompressing intermediate states:
  $$A_{t, s, h} = \frac{\tilde{Q}_{t, h} c_s^T + Q_{t, h}^{pe} (K_s^{pe})^T}{\sqrt{d_h + d_{pe}}}$$
- **Hybrid 3:1 Attention Span**: 18 Sliding Window Attention (SWA, $W=1024$) layers and 6 Global GSLA layers.

### 2. Heterogeneous Factorized MoE (HFK-MoE)
- **Static Shared Anchor**: Dense SwiGLU FFN with $d_{ffn} = 4,096$ executed for **100% of tokens** without gating. This prevents representation collapse and retains base syntactic reasoning.
- **Dynamic Low-Rank Experts**: 16 experts factorized with low rank $r = 384$:
  $$E_k(x) = (\text{swish}(x U_{gate, k}) \odot (x U_{up, k})) V_{down, k}$$
- **Top-4 Routing**: Softmax-normalized Top-4 gating over the 16 experts.
- **Auxiliary Load-Balancing Loss**: Switch Transformer-style loss ($\alpha_{aux} = 0.01$) ensuring uniform expert utilization without capacity drop.

### 3. Stabilization & Global Residual Highway
- **Global Highway**: Preserves early lexical signal across deep layers:
  $$x_{final} = x_{24} + \tanh(\alpha) \cdot \text{RMSNorm}(x_0)$$
  where $\alpha$ is a learned scalar parameter initialized to $0.05$.
- **Tied Weights**: $W_{head} = W_{embed} \in \mathbb{R}^{49152 \times 2048}$.
- **Logit Soft-Capping**: Prevents logit divergence and extreme probabilities:
  $$\text{logits} = 30.0 \cdot \tanh(\text{logits}_{raw} / 30.0)$$

---

## 📁 Repository Structure

```
├── nool_alpha_kaggle_training.ipynb     # Pre-training notebook (Stage 1)
├── nool_alpha_100m_sft_training.ipynb   # Instruction SFT notebook (Stage 2)
├── nool_alpha_100m_reasoning_sft.ipynb  # Deep Reasoning SFT notebook (Stage 2.5)
├── nool_alpha_100m_bridge_chat.ipynb    # Bilingual Bridge & Daily Chat notebook (Stage 3)
├── nool_alpha/
│   ├── __init__.py                      # Package exports
│   ├── config.py                        # NoolAlphaConfig (full_1_5b & nool_100m)
│   ├── model.py                         # PyTorch implementation of GSLA & HFK-MoE
│   ├── dataset.py                       # Pre-training streaming dataset pipeline
│   ├── sft_dataset.py                   # SFT instruction dataset
│   ├── reasoning_dataset.py             # 7-stream reasoning dataset with rolling reservoir
│   ├── bridge_chat_dataset.py           # Bilingual OPUS + UltraChat + Indonesian dialogue
│   ├── reasoning_train.py               # 4.5h Deep Reasoning training engine
│   ├── bridge_chat_train.py             # 3.5h Bilingual Bridge & Chat training engine
│   └── train.py                         # Pre-training engine
├── benchmark_intelligence_peers.py      # Intelligence benchmark vs GPT-2 & SmolLM-135M
├── BENCHMARK_INTELLIGENCE_100M.md       # Empirical intelligence benchmark report
├── generate_bridge_chat_notebook.py     # Stage 3 notebook generator script
├── generate_reasoning_notebook.py       # Stage 2.5 notebook generator script
└── README.md
```

---

## ⚡ How to Train on Kaggle (3–5 Hours)

### Step 1: Upload the Notebook to Kaggle
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. In the top menu, select **File** -> **Import Notebook**.
3. Upload `nool_alpha_kaggle_training.ipynb` from this repository.

### Step 2: Configure Accelerator
1. In the right-hand settings panel:
   - **Accelerator**: Select **GPU T4 x 2** or **GPU P100**.
   - **Internet**: Ensure **Internet On** is checked (needed to stream Hugging Face datasets).

### Step 3: Run Training & Resume from Checkpoint
- **Auto-Detection**: Jika Anda telah meng-upload dataset checkpoint (seperti `/kaggle/input/datasets/chenstillstude/nool-cp/best_checkpoint.pt` atau `/kaggle/input/nool-cp/best_checkpoint.pt`), notebook akan **otomatis mendeteksi dan memuat bobot model beserta step terakhir (misal Step 1750)**!
- **Phase 2 Continuation**: Training akan otomatis melanjutkan dari Step 1750 menuju step target berikutnya (misal +4.500 step ke step 6.250).
- **Probes dengan Repetition Penalty**: Menggunakan decoding $T=0.5$ dan *repetition penalty* $1.2$ agar output bahasa Indonesia, Inggris, dan Python semakin minim repetisi dan tajam.
- **Auto-Save**: Checkpoint baru akan disimpan secara berkala ke `/kaggle/working/checkpoints/` (`best_checkpoint.pt` dan `nool_alpha_100m_final.pt`).

---

## 🧠 Tahap 2.5: Deep Reasoning & Thought-Chain SFT (4.5 Jam)

Notebook `nool_alpha_100m_reasoning_sft.ipynb` melatih model dengan jejak penalaran mendalam (*DeepSeek-R1 style `<think>` traces*):
- **7 Dataset Reasoning**: `cosmopedia-100k`, `Code-Feedback`, `OpenMathInstruct-1`, `R1-Distill-SFT`, `OpenThoughts-114k`, `Mixture-of-Thoughts`, dan `Math-Reasoning`.
- **Zero-OOM Rolling Reservoir**: Buffer rolling 128 sampel (<5 MB RAM) mencegah memory crash pada Kaggle Host RAM.
- **Label Masking**: Backpropagation hanya aktif pada jejak pemikiran `<think>` dan solusi final.

---

## 🌐 Tahap 3: Bilingual Bridge (En <-> Id) & Everyday Natural Conversation (3.5 Jam)

Notebook `nool_alpha_100m_bridge_chat.ipynb` melatih model sebagai asisten percakapan dwibahasa dan obrolan natural sehari-hari:
- **Dataset Blend**:
  - 🌐 **35% OPUS Translation (`kaitchup/opus-Indonesian-to-English` & `hfxunlp/opus-100`)**: Penyelarasan semantik dwibahasa bolak-balik (Inggris $\leftrightarrow$ Indonesia).
  - 💬 **35% UltraChat-200k (`HuggingFaceH4/ultrachat_200k`)**: Dialog multi-turn alami bahasa Inggris untuk percakapan sehari-hari yang luwes dan ramah.
  - 🇮🇩 **30% Alpaca Indonesian Dialogue (`FreedomIntelligence/alpaca-gpt4-indonesian`)**: Interaksi tanya-jawab santun dan luwes dalam bahasa Indonesia.
- **Budget Waktu**: 3.5 Jam (12.600 detik) dengan auto-save checkpoint.
- **Safetensors Export**: Dilengkapi fix `.clone().contiguous().cpu()` untuk mengekspor tied weights tanpa memory-sharing duplicate errors.

### Menjalankan Stage 3 via CLI:
```bash
python nool_alpha/bridge_chat_train.py \
  --checkpoint exported_reasoning_model/model.safetensors \
  --hours 3.5 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 1e-4
```


## 🖥️ Local / Cluster Training

You can also run pre-training from the command line:

```bash
# 1. Install dependencies
pip install torch datasets transformers accelerate matplotlib

# 2. Run unit tests
python tests/test_model.py

# 3. Launch time-budgeted pre-training (e.g. 4 hours)
python -m nool_alpha.train --hours 4.0 --dataset roneneldan/TinyStories --batch_size 8 --grad_accum 4
```

### Switching Datasets
To train on FineWeb-Edu instead of TinyStories:
```python
# In nool_alpha_kaggle_training.ipynb or CLI:
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
```

---

## 🌐 Multi-Domain Training: English, Indonesian & Code

The pre-training pipeline streams an interleaved corpus across three distinct domains from Hugging Face:

1. 🇬🇧 **English (40%)**: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) or [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) for strong reasoning and grammatical foundations.
2. 🇮🇩 **Indonesian (40%)**: [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.id`) covering diverse encyclopedic Indonesian articles (sejarah, sains, budaya, geografi, dll).
3. 💻 **Coding (20%)**: [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) and [`m-a-p/CodeFeedback-Filtered-Instruction`](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) providing Python syntax, algorithms, and problem-solving instructions.

- **Continuous Token Packing**: Packs continuous text into exact sequence chunks ($512$ or $1024$ tokens) with $0\%$ padding overhead.
- **Zero Local Disk Footprint**: Fully streamed in real-time from Hugging Face Hub, fitting within Kaggle's 20GB local disk constraint.

---

## 📊 Model Presets & Scaling

| Specification | `full_1_5b` (Blueprint) | `nool_100m` (Default) |
|---|---|---|
| **Total Parameters** | ~1.58 Billion | ~111 Million (~100M Tier) |
| **Active Parameters** | ~0.95 Billion (60%) | ~97.9 Million (~88%) |
| **Hidden Dimension ($d_{model}$)** | 2,048 | 768 |
| **Number of Layers** | 24 | 10 |
| **Query Heads ($H_q$)** | 16 ($d_h=128$) | 12 ($d_h=64$) |
| **Latent Content Cache ($d_c$)** | 448 | 192 |
| **Decoupled RoPE Key ($d_{pe}$)** | 64 | 32 |
| **KV Cache per Token** | 512 | 224 |
| **Shared SwiGLU FFN** | 4,096 | 1,536 |
| **Dynamic Experts** | 16 (Top-4 active) | 8 (Top-2 active) |
| **Factorized Rank ($r$)** | 384 | 96 |
| **RoPE Base ($\Theta$)** | 500,000 | 500,000 |
| **Logit Soft-Cap** | 30.0 | 30.0 |
| **Training Budget** | 64× H100 (Cluster) | 1× T4 / P100 (3–5 Hours on Kaggle) |

