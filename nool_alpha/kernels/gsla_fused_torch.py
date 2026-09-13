"""
Fused & Memory-Efficient PyTorch Implementation of Grouped-Subspace Latent Attention (GSLA).
Designed to eliminate O(S^2) intermediate attention matrix allocation via block tiling and online softmax,
providing FlashAttention-level memory efficiency while supporting GSLA's dual-subspace scoring.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def gsla_reference_attention(
    q_tilde: torch.Tensor,
    q_pe: torch.Tensor,
    c_t: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Exact mathematical reference for GSLA:
      A_t,s,h = (Q_tilde @ c_s^T + Q_pe @ (K_s^pe)^T) / sqrt(d_h + d_pe)
      O_t,h = softmax(A) @ V
    """
    d_pe = q_pe.shape[-1]
    d_h = v.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d_h + d_pe)

    # Content score: q_tilde (B, H, Sq, dc), c_t (B, Sk, dc) -> (B, H, Sq, Sk)
    content_scores = torch.einsum("bhqd,bkd->bhqk", q_tilde, c_t)
    
    # Position score: q_pe (B, H, Sq, dpe), k_pe (B, Sk, dpe) -> (B, H, Sq, Sk)
    pos_scores = torch.einsum("bhqd,bkd->bhqk", q_pe, k_pe)
    
    scores = (content_scores + pos_scores) * scale

    if is_causal:
        seq_q = q_tilde.shape[2]
        seq_k = c_t.shape[1]
        mask = torch.triu(torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_tilde.dtype)
    # attn_weights: (B, H, Sq, Sk), v: (B, H, Sk, dh) -> (B, H, Sq, dh)
    output = torch.matmul(attn_weights, v)
    return output


def gsla_fused_attention(
    q_tilde: torch.Tensor,
    q_pe: torch.Tensor,
    c_t: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Memory-efficient chunked/tiled GSLA attention with online softmax in PyTorch.
    Eliminates O(S^2) intermediate memory allocation:
      - Memory overhead is O(chunk_size * seq_len) instead of O(seq_len^2).
      - Mathematically identical to FlashAttention-2 online softmax computation.
    """
    batch_size, num_heads, seq_q, d_c = q_tilde.shape
    _, seq_k, d_pe = k_pe.shape
    d_h = v.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(d_h + d_pe)

    # For short sequences, direct execution is faster
    if seq_q <= chunk_size and seq_k <= chunk_size:
        return gsla_reference_attention(q_tilde, q_pe, c_t, k_pe, v, is_causal=is_causal, scale=scale)

    output = torch.zeros((batch_size, num_heads, seq_q, d_h), dtype=q_tilde.dtype, device=q_tilde.device)

    # Chunk along the query dimension to bound intermediate activation memory
    for q_start in range(0, seq_q, chunk_size):
        q_end = min(q_start + chunk_size, seq_q)
        q_chunk_len = q_end - q_start

        q_t_chunk = q_tilde[:, :, q_start:q_end, :] # (B, H, q_chunk, d_c)
        q_p_chunk = q_pe[:, :, q_start:q_end, :]    # (B, H, q_chunk, d_pe)

        # Online softmax states for this query chunk
        m_i = torch.full((batch_size, num_heads, q_chunk_len, 1), float("-inf"), dtype=torch.float32, device=q_tilde.device)
        l_i = torch.zeros((batch_size, num_heads, q_chunk_len, 1), dtype=torch.float32, device=q_tilde.device)
        acc = torch.zeros((batch_size, num_heads, q_chunk_len, d_h), dtype=torch.float32, device=q_tilde.device)

        # In causal mode, we only need keys up to q_end
        k_limit = q_end if is_causal else seq_k

        for k_start in range(0, k_limit, chunk_size):
            k_end = min(k_start + chunk_size, k_limit)
            
            c_chunk = c_t[:, k_start:k_end, :]     # (B, k_chunk, d_c)
            k_p_chunk = k_pe[:, k_start:k_end, :]  # (B, k_chunk, d_pe)
            v_chunk = v[:, :, k_start:k_end, :]    # (B, H, k_chunk, d_h)

            # 1. Dual-subspace scoring
            s_content = torch.einsum("bhqd,bkd->bhqk", q_t_chunk, c_chunk)
            s_pos = torch.einsum("bhqd,bkd->bhqk", q_p_chunk, k_p_chunk)
            s_chunk = (s_content + s_pos) * scale

            # 2. Causal masking within chunk
            if is_causal:
                # Query index: [q_start, q_end), Key index: [k_start, k_end)
                q_idx = torch.arange(q_start, q_end, device=q_tilde.device).view(1, 1, -1, 1)
                k_idx = torch.arange(k_start, k_end, device=q_tilde.device).view(1, 1, 1, -1)
                causal_mask = k_idx > q_idx
                s_chunk = s_chunk.masked_fill(causal_mask, float("-inf"))

            # 3. Online Softmax update (FlashAttention formula)
            m_ij = torch.max(s_chunk, dim=-1, keepdim=True).values.to(torch.float32)
            m_new = torch.maximum(m_i, m_ij)
            alpha = torch.exp(m_i - m_new)
            
            p_chunk = torch.exp(s_chunk.to(torch.float32) - m_new)
            
            # Rescale previous accumulator
            acc = acc * alpha
            l_i = l_i * alpha + torch.sum(p_chunk, dim=-1, keepdim=True)
            m_i = m_new

            # 4. Accumulate weighted values
            p_chunk_cast = p_chunk.to(v_chunk.dtype)
            acc += torch.matmul(p_chunk_cast, v_chunk).to(torch.float32)

        # Normalize by sum of exponentials
        out_chunk = acc / torch.clamp(l_i, min=1e-8)
        output[:, :, q_start:q_end, :] = out_chunk.to(q_tilde.dtype)

    return output


class GSLAPyTorchFused(nn.Module):
    """Module wrapper for Fused GSLA Attention."""
    def __init__(self, chunk_size: int = 256):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, q_tilde, q_pe, c_t, k_pe, v, is_causal=True, scale=None):
        return gsla_fused_attention(
            q_tilde, q_pe, c_t, k_pe, v,
            is_causal=is_causal, scale=scale,
            chunk_size=self.chunk_size
        )


def verify_kernel_parity(
    batch_size: int = 2,
    num_heads: int = 12,
    seq_len: int = 512,
    d_c: int = 192,
    d_pe: int = 32,
    d_h: int = 64,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[bool, float]:
    """
    Verifies numerical equivalence between Reference GSLA and Fused GSLA.
    Returns: (is_valid, max_absolute_difference)
    """
    torch.manual_seed(42)
    q_tilde = torch.randn(batch_size, num_heads, seq_len, d_c, device=device, dtype=dtype)
    q_pe = torch.randn(batch_size, num_heads, seq_len, d_pe, device=device, dtype=dtype)
    c_t = torch.randn(batch_size, seq_len, d_c, device=device, dtype=dtype)
    k_pe = torch.randn(batch_size, seq_len, d_pe, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, d_h, device=device, dtype=dtype)

    ref_out = gsla_reference_attention(q_tilde, q_pe, c_t, k_pe, v, is_causal=True)
    fused_out = gsla_fused_attention(q_tilde, q_pe, c_t, k_pe, v, is_causal=True, chunk_size=128)

    max_diff = torch.max(torch.abs(ref_out - fused_out)).item()
    is_valid = max_diff < 1e-4
    return is_valid, max_diff
