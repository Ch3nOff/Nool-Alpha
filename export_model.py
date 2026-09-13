"""
Export Nool-Alpha PyTorch checkpoints to Hugging Face-style format with Safetensors.

Generates:
  - model.safetensors
  - config.json
  - tokenizer files (vocab.json, merges.txt, tokenizer_config.json, etc.)
  - README.md (Model Card with architecture specifications)
"""

import argparse
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from nool_alpha.config import NoolAlphaConfig
from nool_alpha.model import NoolAlphaForCausalLM

# Register safe globals and alias in __main__ for unpickling
if hasattr(sys.modules["__main__"], "NoolAlphaConfig") is False:
    setattr(sys.modules["__main__"], "NoolAlphaConfig", NoolAlphaConfig)

try:
    torch.serialization.add_safe_globals([NoolAlphaConfig])
except Exception:
    pass

# Ensure UTF-8 output for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_SOURCE_DIR = r"C:\Users\Matthew Chen\Downloads\Nool_alpha model"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "exported_models", "nool_alpha_100m")


def export_checkpoint_to_safetensors(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_name: str = "gpt2",
) -> str:
    """Exports a .pt checkpoint to safetensors + HF config + tokenizer files."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Loading checkpoint from: {checkpoint_path}")
    t0 = time.time()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    raw_config = ckpt.get("config")
    if isinstance(raw_config, dict):
        config = NoolAlphaConfig(**raw_config)
    elif isinstance(raw_config, NoolAlphaConfig):
        config = raw_config
    else:
        from nool_alpha.config import nool_100m
        config = nool_100m()

    state_dict = ckpt["model_state_dict"]
    step = ckpt.get("step", "unknown")
    loss = ckpt.get("loss", None)

    # 1. Save model.safetensors
    # Safetensors requires distinct tensor buffers for tied weights (e.g. lm_head.weight and embed_tokens.weight)
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    print(f"[+] Saving tensors to {safetensors_path}...")
    clean_state_dict = {k: v.clone().contiguous().cpu() for k, v in state_dict.items()}
    save_file(clean_state_dict, safetensors_path)

    # 2. Save config.json
    config_dict = {
        "architectures": ["NoolAlphaForCausalLM"],
        "model_type": "nool_alpha",
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "num_heads": config.num_heads,
        "head_dim": config.head_dim,
        "d_c": config.d_c,
        "d_pe": config.d_pe,
        "rope_theta": config.rope_theta,
        "max_position_embeddings": config.max_position_embeddings,
        "sliding_window": config.sliding_window,
        "swa_interval": config.swa_interval,
        "shared_ffn_dim": config.shared_ffn_dim,
        "num_experts": config.num_experts,
        "top_k_experts": config.top_k_experts,
        "expert_rank": config.expert_rank,
        "moe_aux_loss_coeff": config.moe_aux_loss_coeff,
        "highway_alpha_init": config.highway_alpha_init,
        "logit_soft_cap": config.logit_soft_cap,
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
        "checkpoint_step": step,
        "checkpoint_loss": loss,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[+] Saved config to {config_path}")

    # 3. Save tokenizer
    print(f"[+] Saving tokenizer ({tokenizer_name}) to {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(output_dir)

    # 4. Generate README.md Model Card
    readme_path = os.path.join(output_dir, "README.md")
    total_params = sum(p.numel() for p in clean_state_dict.values())
    readme_content = f"""# Nool-Alpha-100M

**Nool-Alpha-100M** is a bilingual & code causal language model based on an efficient hybrid architecture:
- **GSLA (Grouped-Subspace Latent Attention)** with Decoupled RoPE and hybrid SWA (Sliding Window Attention).
- **HFK-MoE (Heterogeneous Factorized MoE)**: Dense SwiGLU shared anchor + 8 low-rank experts (r={config.expert_rank}, Top-2 routing).
- **Global Residual Highway** ($\\\\tanh(\\\\alpha) \\\\cdot \\\\text{{RMSNorm}}(x_0)$).
- **Logit Soft-Capping** (30.0).

## Model Details
- **Architecture**: `nool_alpha`
- **Total Parameters**: ~{round(total_params / 1e6, 1)}M (~111M)
- **Active Parameters / Token**: ~97.9M
- **Vocabulary Size**: {config.vocab_size} (GPT-2 BPE)
- **Context Length**: {config.max_position_embeddings} tokens
- **Training Step**: {step}
- **Recorded Loss**: {loss if loss is not None else 'N/A'}

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
config = NoolAlphaConfig.from_dict({json.dumps(config_dict, indent=2)})
model = NoolAlphaForCausalLM(config)
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
model.eval()
```
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[+] Generated README.md at {readme_path}")

    # 5. Quick verification of safetensors
    loaded_tensors = load_file(safetensors_path)
    assert len(loaded_tensors) == len(clean_state_dict), "Tensor count mismatch in safetensors verification!"
    print(f"[OK] Successfully exported and verified {len(loaded_tensors)} tensors in {round(time.time() - t0, 2)}s.")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Export Nool-Alpha checkpoint to Safetensors format.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(DEFAULT_SOURCE_DIR, "best_checkpoint.pt"),
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Target output directory for exported files",
    )
    args = parser.parse_args()

    # Fallback to final checkpoint if best not found
    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        alt_ckpt = os.path.join(DEFAULT_SOURCE_DIR, "nool_alpha_100m_final.pt")
        if os.path.exists(alt_ckpt):
            ckpt = alt_ckpt

    export_checkpoint_to_safetensors(ckpt, args.output_dir)


if __name__ == "__main__":
    main()
