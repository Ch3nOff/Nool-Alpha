from dataclasses import dataclass
from typing import Optional

@dataclass
class NoolAlphaConfig:
    """
    Configuration for Nool-Alpha LLM.
    Implements Grouped-Subspace Latent Attention (GSLA) with Decoupled RoPE
    and Heterogeneous Factorized MoE (HFK-MoE).
    """
    # Architecture Dimensions
    vocab_size: int = 49152
    d_model: int = 2048
    n_layers: int = 24
    num_heads: int = 16          # H_q: 16 query heads
    head_dim: int = 128          # d_h: 128 (16 * 128 = 2048 = d_model)
    
    # GSLA (Grouped-Subspace Latent Attention) Parameters
    d_c: int = 448               # Latent value/content cache dimension (W_DK down-projection)
    d_pe: int = 64               # Decoupled RoPE Key dimension (W_pos down-projection)
    rope_theta: float = 500000.0 # Positional frequency base
    max_position_embeddings: int = 8192
    
    # Hybrid Attention Span Structure (3:1 ratio)
    # 18 SWA layers (window size W=1024), 6 global GSLA layers
    sliding_window: int = 1024
    swa_interval: int = 4        # Every 4th layer (layer_idx % 4 == 3) is Global Attention; others SWA
    
    # HFK-MoE (Heterogeneous Factorized MoE) Parameters
    shared_ffn_dim: int = 4096   # Static dense SwiGLU anchor dimension
    num_experts: int = 16        # Dynamic path: 16 experts
    top_k_experts: int = 4       # Top-4 experts routed per token
    expert_rank: int = 384       # r = 384 low-rank factorized projection dimension
    moe_aux_loss_coeff: float = 0.01  # Auxiliary load balancing loss coefficient
    
    # Stabilization & Global Residual Highway
    highway_alpha_init: float = 0.05  # x_final = x_L + tanh(alpha) * RMSNorm(x_0)
    logit_soft_cap: float = 30.0      # 30.0 * tanh(logits / 30.0)
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True  # Weight tying between embedding and LM head
    dropout: float = 0.0

    @classmethod
    def full_1_5b(cls) -> "NoolAlphaConfig":
        """Exact 1.58B total / 0.95B active parameter blueprint configuration."""
        return cls()

    @classmethod
    def nool_100m(cls, vocab_size: int = 50257) -> "NoolAlphaConfig":
        """
        Scaled ~100M parameter model retaining 100% of Nool-Alpha architectural fidelity:
          - GSLA (Grouped-Subspace Latent Attention with Decoupled RoPE theta=500k)
          - HFK-MoE (Static SwiGLU Shared Anchor + Dynamic Low-Rank Experts)
          - Global Residual Highway (tanh(alpha) * RMSNorm(x0))
          - Tied Weights & Logit Soft-Capping (30 * tanh(logits/30))
        Total params: ~98M-105M (Active params: ~65M-72M).
        """
        return cls(
            vocab_size=vocab_size,
            d_model=768,
            n_layers=10,
            num_heads=12,
            head_dim=64,
            d_c=192,
            d_pe=32,
            rope_theta=500000.0,
            max_position_embeddings=2048,
            sliding_window=512,
            swa_interval=4,
            shared_ffn_dim=1536,
            num_experts=8,
            top_k_experts=2,
            expert_rank=96,
            moe_aux_loss_coeff=0.01,
            highway_alpha_init=0.05,
            logit_soft_cap=30.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
        )

    @classmethod
    def kaggle_3_5h(cls, vocab_size: int = 49152) -> "NoolAlphaConfig":
        """
        Scaled-down configuration preserving the exact GSLA + HFK-MoE mechanics,
        ideal for training from scratch within 3-5 hours on a Kaggle T4 or P100 GPU (16GB VRAM).
        ~130M total parameters, highly responsive with high MFU throughput.
        """
        return cls(
            vocab_size=vocab_size,
            d_model=768,
            n_layers=12,
            num_heads=12,
            head_dim=64,         # 12 * 64 = 768
            d_c=192,             # Latent content cache
            d_pe=32,             # Decoupled RoPE subspace
            rope_theta=500000.0,
            max_position_embeddings=2048,
            sliding_window=512,
            swa_interval=4,      # 3 SWA : 1 Global attention
            shared_ffn_dim=1536, # SwiGLU shared anchor
            num_experts=8,       # 8 dynamic experts
            top_k_experts=2,     # Top-2 active per token
            expert_rank=128,     # Low-rank factorized dimension
            moe_aux_loss_coeff=0.01,
            highway_alpha_init=0.05,
            logit_soft_cap=30.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True
        )

    @classmethod
    def smoke_test(cls) -> "NoolAlphaConfig":
        """Minimal configuration for instant unit testing and sanity checks."""
        return cls(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            num_heads=4,
            head_dim=32,
            d_c=32,
            d_pe=16,
            rope_theta=10000.0,
            max_position_embeddings=256,
            sliding_window=64,
            swa_interval=2,
            shared_ffn_dim=256,
            num_experts=4,
            top_k_experts=2,
            expert_rank=32,
            moe_aux_loss_coeff=0.01,
            highway_alpha_init=0.05,
            logit_soft_cap=30.0
        )
