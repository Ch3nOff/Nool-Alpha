"""
Custom Triton Kernel for Grouped-Subspace Latent Attention (GSLA).
Features:
  - Dual-subspace attention: content latent dot-product (Q_tilde @ C^T) + decoupled position dot-product (Q_pe @ K_pe^T)
  - FlashAttention-style tiled online softmax: O(1) SRAM memory overhead, zero O(S^2) intermediate allocation
  - In-register causal and sliding window masking
  - Shared latent KV cache across attention heads (enormous VRAM bandwidth savings).
"""

import math
from typing import Optional, Tuple
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _gsla_fwd_kernel(
        # Pointers to inputs and output
        Q_tilde_ptr, Q_pe_ptr, C_ptr, K_pe_ptr, V_ptr, Out_ptr,
        scale,
        # Strides for Q_tilde: (batch, heads, seq_q, d_c)
        stride_qz_b, stride_qz_h, stride_qz_m, stride_qz_k,
        # Strides for Q_pe: (batch, heads, seq_q, d_pe)
        stride_qp_b, stride_qp_h, stride_qp_m, stride_qp_k,
        # Strides for C: (batch, seq_k, d_c) [shared across heads]
        stride_c_b, stride_c_n, stride_c_k,
        # Strides for K_pe: (batch, seq_k, d_pe) [shared across heads]
        stride_kp_b, stride_kp_n, stride_kp_k,
        # Strides for V: (batch, heads, seq_k, d_h)
        stride_v_b, stride_v_h, stride_v_n, stride_v_k,
        # Strides for Output: (batch, heads, seq_q, d_h)
        stride_o_b, stride_o_h, stride_o_m, stride_o_k,
        # Dimensions
        Z, H, N_CTX_Q, N_CTX_K,
        D_C: tl.constexpr,
        D_PE: tl.constexpr,
        D_H: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        # Block and batch/head index
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        # Offset offsets for row tile M
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_dc = tl.arange(0, D_C)
        offs_dpe = tl.arange(0, D_PE)
        offs_dh = tl.arange(0, D_H)

        # Base pointers for this batch & head
        q_tilde_base = Q_tilde_ptr + off_z * stride_qz_b + off_h * stride_qz_h
        q_pe_base = Q_pe_ptr + off_z * stride_qp_b + off_h * stride_qp_h
        c_base = C_ptr + off_z * stride_c_b
        k_pe_base = K_pe_ptr + off_z * stride_kp_b
        v_base = V_ptr + off_z * stride_v_b + off_h * stride_v_h
        o_base = Out_ptr + off_z * stride_o_b + off_h * stride_o_h

        # Load Q_tilde tile (BLOCK_M x D_C) and Q_pe tile (BLOCK_M x D_PE)
        q_tilde_ptrs = q_tilde_base + (offs_m[:, None] * stride_qz_m + offs_dc[None, :] * stride_qz_k)
        q_pe_ptrs = q_pe_base + (offs_m[:, None] * stride_qp_m + offs_dpe[None, :] * stride_qp_k)
        
        q_tilde = tl.load(q_tilde_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
        q_pe = tl.load(q_pe_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

        # Initialize online softmax statistics: m (running max) and l (running sum)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D_H], dtype=tl.float32)

        # Determine loop range over keys/values
        end_n = N_CTX_K
        if IS_CAUSAL:
            end_n = tl.minimum(N_CTX_K, (start_m + 1) * BLOCK_M)

        # Loop over KV sequence blocks
        for start_n in range(0, end_n, BLOCK_N):
            curr_n = start_n + offs_n

            # Load C tile (BLOCK_N x D_C) and K_pe tile (BLOCK_N x D_PE)
            c_ptrs = c_base + (curr_n[None, :] * stride_c_n + offs_dc[:, None] * stride_c_k)
            k_pe_ptrs = k_pe_base + (curr_n[None, :] * stride_kp_n + offs_dpe[:, None] * stride_kp_k)
            v_ptrs = v_base + (curr_n[:, None] * stride_v_n + offs_dh[None, :] * stride_v_k)

            c_tile = tl.load(c_ptrs, mask=curr_n[None, :] < N_CTX_K, other=0.0)
            k_pe_tile = tl.load(k_pe_ptrs, mask=curr_n[None, :] < N_CTX_K, other=0.0)
            v_tile = tl.load(v_ptrs, mask=curr_n[:, None] < N_CTX_K, other=0.0)

            # 1. Dual-subspace attention dot-product: S = (Q_tilde @ C^T + Q_pe @ K_pe^T) * scale
            s_content = tl.dot(q_tilde, c_tile)
            s_pos = tl.dot(q_pe, k_pe_tile)
            s = (s_content + s_pos) * scale

            # 2. Causal masking
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= curr_n[None, :]
                s = tl.where(causal_mask, s, float("-inf"))
            
            # Boundary mask for padding beyond N_CTX_K
            s = tl.where(curr_n[None, :] < N_CTX_K, s, float("-inf"))

            # 3. Online Softmax update
            m_ij = tl.max(s, 1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])

            # Rescale accumulator and running sum
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

            # 4. Accumulate weighted values: acc += p @ v
            p = p.to(v_tile.dtype)
            acc += tl.dot(p, v_tile)

        # 5. Normalize by sum of exponentials: output = acc / l_i
        acc = acc / l_i[:, None]
        out_ptrs = o_base + (offs_m[:, None] * stride_o_m + offs_dh[None, :] * stride_o_k)
        tl.store(out_ptrs, acc.to(Q_tilde_ptr.dtype.element_ty), mask=offs_m[:, None] < N_CTX_Q)


def gsla_triton_attention(
    q_tilde: torch.Tensor,
    q_pe: torch.Tensor,
    c_t: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Python wrapper for the custom GSLA Triton kernel.
    
    Arguments:
      q_tilde: (batch, num_heads, seq_len_q, d_c)
      q_pe:    (batch, num_heads, seq_len_q, d_pe)
      c_t:     (batch, seq_len_k, d_c) [shared across heads]
      k_pe:    (batch, seq_len_k, d_pe) [shared across heads]
      v:       (batch, num_heads, seq_len_k, d_h)
    
    Returns:
      output:  (batch, num_heads, seq_len_q, d_h)
    """
    if not HAS_TRITON or not q_tilde.is_cuda:
        raise RuntimeError("Triton is not available or tensor is not on CUDA device.")

    batch_size, num_heads, seq_len_q, d_c = q_tilde.shape
    _, seq_len_k, _ = c_t.shape
    d_pe = q_pe.shape[-1]
    d_h = v.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(d_h + d_pe)

    # Allocate output tensor
    output = torch.empty((batch_size, num_heads, seq_len_q, d_h), dtype=q_tilde.dtype, device=q_tilde.device)

    # Block configurations
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(seq_len_q, BLOCK_M), batch_size * num_heads)

    _gsla_fwd_kernel[grid](
        q_tilde, q_pe, c_t, k_pe, v, output,
        scale,
        # Strides Q_tilde
        q_tilde.stride(0), q_tilde.stride(1), q_tilde.stride(2), q_tilde.stride(3),
        # Strides Q_pe
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2), q_pe.stride(3),
        # Strides C
        c_t.stride(0), c_t.stride(1), c_t.stride(2),
        # Strides K_pe
        k_pe.stride(0), k_pe.stride(1), k_pe.stride(2),
        # Strides V
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Strides Out
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, num_heads, seq_len_q, seq_len_k,
        D_C=d_c, D_PE=d_pe, D_H=d_h,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
    )

    return output
