# Nool-Alpha-100M

**Nool-Alpha-100M** is a bilingual & code causal language model based on an efficient hybrid architecture:
- **GSLA (Grouped-Subspace Latent Attention)** with Decoupled RoPE and hybrid SWA (Sliding Window Attention).
- **HFK-MoE (Heterogeneous Factorized MoE)**: Dense SwiGLU shared anchor + 8 low-rank experts (r=96, Top-2 routing).
- **Global Residual Highway** ($\\tanh(\\alpha) \\cdot \\text{RMSNorm}(x_0)$).
- **Logit Soft-Capping** (30.0).

## Model Details
- **Architecture**: `nool_alpha`
- **Total Parameters**: ~149.8M (~111M)
- **Active Parameters / Token**: ~97.9M
- **Vocabulary Size**: 50257 (GPT-2 BPE)
- **Context Length**: 2048 tokens
- **Training Step**: 1339
- **Recorded Loss**: 2.7460535764694214

## Files Included
- `model.safetensors`: Model weights in safe, zero-copy format.
- `config.json`: Complete architectural hyper-parameters.
- Tokenizer files: `vocab.json`, `merges.txt`, `tokenizer.json`, etc.

## Quick Start (PyTorch)
```python
import torch
from safetensors.torch import load_file
from nool_alpha.config import NoolAlphaConfig
from nool_alpha.model import NoolAlphaForCausalLM

# Load configuration and weights
config = NoolAlphaConfig.from_dict({
  "architectures": [
    "NoolAlphaForCausalLM"
  ],
  "model_type": "nool_alpha",
  "vocab_size": 50257,
  "d_model": 768,
  "n_layers": 10,
  "num_heads": 12,
  "head_dim": 64,
  "d_c": 192,
  "d_pe": 32,
  "rope_theta": 500000.0,
  "max_position_embeddings": 2048,
  "sliding_window": 512,
  "swa_interval": 4,
  "shared_ffn_dim": 1536,
  "num_experts": 8,
  "top_k_experts": 2,
  "expert_rank": 96,
  "moe_aux_loss_coeff": 0.01,
  "highway_alpha_init": 0.05,
  "logit_soft_cap": 30.0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": true,
  "checkpoint_step": 1339,
  "checkpoint_loss": 2.7460535764694214
})
model = NoolAlphaForCausalLM(config)
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
model.eval()
```
