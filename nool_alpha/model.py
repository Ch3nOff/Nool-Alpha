import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import NoolAlphaConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learned scale parameter."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS along the last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class DecoupledRotaryEmbedding(nn.Module):
    """
    Decoupled Rotary Position Embedding for dedicated positional subspace d_pe.
    Base frequency theta = 500,000 as specified in Nool-Alpha blueprint.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_position_embeddings
        self.base = base
        
        # Precompute inverse frequencies for d_pe / 2 pairs
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cos_sin_cache(self.max_seq_len)

    def _build_cos_sin_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, num_heads, d_pe) or (batch, seq_len, d_pe)
        returns cos, sin slices of shape matching seq_len
        """
        if seq_len > self.cos_cached.shape[0]:
            self._build_cos_sin_cache(seq_len)
        cos = self.cos_cached[:seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len, :].to(dtype=x.dtype, device=x.device)
        return cos, sin

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to tensor x:
        x: (batch, seq_len, dim) or (batch, seq_len, num_heads, dim)
        cos, sin: (seq_len, dim)
        """
        if x.ndim == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        elif x.ndim == 4:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        return (x * cos) + (self._rotate_half(x) * sin)


class GroupedSubspaceLatentAttention(nn.Module):
    """
    Grouped-Subspace Latent Attention (GSLA) with Decoupled RoPE.
    
    Mathematical Formulation:
      - Content compression: c_t = x_t W_DK in R^(d_c)
      - Dedicated position subspace: K_t^pe = RoPE(x_t W_pos) in R^(d_pe)
      - KV cache per token: [c_t || K_t^pe] in R^(d_c + d_pe) = 512
      - Query projection: Q_val in R^(H_q x d_h), Q_pe in R^(H_q x d_pe)
      - Key up-projection absorbed into Query: Q_tilde = Q_val @ W_UK^T in R^(d_c)
      - Attention score: A_t,s,h = (Q_tilde @ c_s^T + Q_pe @ (K_s^pe)^T) / sqrt(d_h + d_pe)
      - Hybrid 3:1 attention span (Sliding Window Attention / Global Attention).
    """
    def __init__(self, config: NoolAlphaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_c = config.d_c
        self.d_pe = config.d_pe
        
        # Scaling factor: sqrt(d_h + d_pe)
        self.scale = 1.0 / math.sqrt(self.head_dim + self.d_pe)
        
        # Hybrid 3:1 attention span: every swa_interval-th layer is Global Attention; others SWA
        self.is_sliding_window = (config.sliding_window is not None) and (
            (layer_idx % config.swa_interval) != (config.swa_interval - 1)
        )
        self.sliding_window = config.sliding_window if self.is_sliding_window else None

        # Content down-projection W_DK: d_model -> d_c
        self.w_dk = nn.Linear(self.d_model, self.d_c, bias=False)
        
        # Position down-projection W_pos: d_model -> d_pe
        self.w_pos = nn.Linear(self.d_model, self.d_pe, bias=False)
        
        # Query projection: d_model -> H_q * (d_h + d_pe)
        self.w_q = nn.Linear(self.d_model, self.num_heads * (self.head_dim + self.d_pe), bias=False)
        
        # Key up-projection W_UK: per-head projection from d_c -> d_h (or absorbed Q_tilde in d_c)
        # We store W_UK of shape (num_heads, head_dim, d_c)
        self.w_uk = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.d_c))
        nn.init.kaiming_uniform_(self.w_uk, a=math.sqrt(5))
        
        # Value up-projection W_UV: per-head projection from d_c -> d_h
        self.w_uv = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.d_c))
        nn.init.kaiming_uniform_(self.w_uv, a=math.sqrt(5))
        
        # Output projection W_O: H_q * d_h -> d_model
        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        
        # Decoupled RoPE for the d_pe subspace
        self.rotary_emb = DecoupledRotaryEmbedding(
            dim=self.d_pe,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Content compression: c_t in R^(d_c)
        c_t = self.w_dk(hidden_states) # (batch, seq_len, d_c)
        
        # 2. Position down-projection and RoPE rotation: K_pe in R^(d_pe)
        k_pe = self.w_pos(hidden_states) # (batch, seq_len, d_pe)
        cos, sin = self.rotary_emb(k_pe, seq_len)
        k_pe = self.rotary_emb.apply_rope(k_pe, cos, sin) # (batch, seq_len, d_pe)
        
        # Manage KV Cache (storing c_t: 448d, k_pe: 64d = 512d total)
        if kv_cache is not None:
            prev_c, prev_k_pe = kv_cache
            c_t = torch.cat([prev_c, c_t], dim=1)
            k_pe = torch.cat([prev_k_pe, k_pe], dim=1)
        
        current_cache = (c_t, k_pe) if use_cache else None
        total_seq_len = c_t.shape[1]

        # 3. Query projection: split into Q_val (d_h) and Q_pe (d_pe)
        q_proj = self.w_q(hidden_states) # (batch, seq_len, num_heads * (head_dim + d_pe))
        q_proj = q_proj.view(batch_size, seq_len, self.num_heads, self.head_dim + self.d_pe)
        q_val = q_proj[..., : self.head_dim] # (batch, seq_len, num_heads, head_dim)
        q_pe = q_proj[..., self.head_dim :]  # (batch, seq_len, num_heads, d_pe)
        
        # Apply RoPE to Query positional subspace
        q_pe = self.rotary_emb.apply_rope(q_pe, cos, sin) # (batch, seq_len, num_heads, d_pe)

        # 4. Matrix Absorption for Content Path:
        # Absorbing W_UK into Query: Q_tilde = Q_val @ W_UK in R^(d_c)
        # q_val: (batch, seq_len, num_heads, head_dim)
        # w_uk: (num_heads, head_dim, d_c)
        # einsum computes: Q_tilde[b, t, h, c] = sum_d (q_val[b, t, h, d] * w_uk[h, d, c])
        q_tilde = torch.einsum("bthd,hdc->bthc", q_val, self.w_uk) # (batch, seq_len, num_heads, d_c)

        # 5. Attention Scores: A = (Q_tilde @ c_s^T + Q_pe @ (K_s^pe)^T) * scale
        # Content score:
        # q_tilde: (batch, seq_len, num_heads, d_c)
        # c_t: (batch, total_seq_len, d_c)
        content_score = torch.einsum("bthc,bsc->bhts", q_tilde, c_t) # (batch, num_heads, seq_len, total_seq_len)
        
        # Position score:
        # q_pe: (batch, seq_len, num_heads, d_pe)
        # k_pe: (batch, total_seq_len, d_pe)
        pos_score = torch.einsum("bthd,bsd->bhts", q_pe, k_pe) # (batch, num_heads, seq_len, total_seq_len)
        
        scores = (content_score + pos_score) * self.scale # (batch, num_heads, seq_len, total_seq_len)

        # 6. Apply Causal & Sliding Window Mask
        # Standard causal mask: cannot attend to future tokens (s > t)
        offset = total_seq_len - seq_len
        q_pos = torch.arange(seq_len, device=scores.device).unsqueeze(1) + offset # (seq_len, 1)
        k_pos = torch.arange(total_seq_len, device=scores.device).unsqueeze(0)    # (1, total_seq_len)
        
        causal_mask = k_pos > q_pos # True where future token
        
        if self.is_sliding_window and self.sliding_window is not None:
            # Sliding window: cannot attend to tokens older than (q_pos - sliding_window)
            window_mask = k_pos < (q_pos - self.sliding_window)
            mask = causal_mask | window_mask
        else:
            mask = causal_mask
            
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        if attention_mask is not None:
            scores = scores + attention_mask

        # 7. Softmax & Value Aggregation
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # Value up-projection:
        # Instead of expanding full V before attention, we can project c_t to V_h = c_t @ W_UV^T
        # c_t: (batch, total_seq_len, d_c)
        # w_uv: (num_heads, head_dim, d_c)
        v_h = torch.einsum("bsc,hdc->bhsd", c_t, self.w_uv) # (batch, num_heads, total_seq_len, head_dim)
        
        # Output: (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v_h)
        
        # Reshape and project through W_O
        # (batch, seq_len, num_heads * head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        output = self.w_o(attn_output)
        
        return output, current_cache


class FactorizedExpert(nn.Module):
    """
    Low-rank factorized expert for HFK-MoE.
    E_k(x) = (swish(x @ U_gate,k) * (x @ U_up,k)) @ V_down,k
    U in R^(d_model x r), V in R^(r x d_model), with rank r=384.
    """
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.u_gate = nn.Linear(d_model, rank, bias=False)
        self.u_up = nn.Linear(d_model, rank, bias=False)
        self.v_down = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation over low-rank subspace
        return self.v_down(F.silu(self.u_gate(x)) * self.u_up(x))


class HeterogeneousFactorizedMoE(nn.Module):
    """
    Heterogeneous Factorized MoE (HFK-MoE).
    
    Structure:
      - Static Path (Shared Anchor):
          FFN_shared(x) = (swish(x @ W_gate) * (x @ W_up)) @ W_down
          Dense SwiGLU running on 100% of tokens with d_ffn = 4,096.
      - Dynamic Path:
          16 Factorized Experts (r = 384).
          Top-4 routing per token with softmax-normalized routing weights.
      - Auxiliary Load-Balancing Loss (Switch Transformer style) to prevent expert collapse.
    """
    def __init__(self, config: NoolAlphaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.shared_ffn_dim = config.shared_ffn_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.expert_rank = config.expert_rank
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        # --- 1. Static Path: Dense SwiGLU Shared Anchor ---
        self.shared_w_gate = nn.Linear(self.d_model, self.shared_ffn_dim, bias=False)
        self.shared_w_up = nn.Linear(self.d_model, self.shared_ffn_dim, bias=False)
        self.shared_w_down = nn.Linear(self.shared_ffn_dim, self.d_model, bias=False)

        # --- 2. Dynamic Path: 16 Factorized Low-Rank Experts ---
        self.experts = nn.ModuleList([
            FactorizedExpert(self.d_model, self.expert_rank) for _ in range(self.num_experts)
        ])
        
        # --- 3. Router / Gating Network ---
        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # (N_tokens, d_model)
        num_tokens = x_flat.shape[0]

        # --- Static Shared Anchor Computation (100% tokens) ---
        shared_out = self.shared_w_down(
            F.silu(self.shared_w_gate(x_flat)) * self.shared_w_up(x_flat)
        )

        # --- Dynamic MoE Gating & Routing ---
        router_logits = self.router(x_flat) # (N_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1) # (N_tokens, num_experts)

        # Select Top-K experts per token
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # Softmax normalize over the selected top-k experts
        topk_weights = F.softmax(topk_weights, dim=-1) # (N_tokens, top_k)

        # --- Dynamic Expert Evaluation ---
        dynamic_out = torch.zeros_like(x_flat)
        
        # Token dispatching across the selected experts
        for k in range(self.top_k):
            # Gather indices for the k-th expert assignment
            expert_indices = topk_indices[:, k] # (N_tokens,)
            weights = topk_weights[:, k].unsqueeze(-1) # (N_tokens, 1)

            # Evaluate each expert on its assigned tokens
            for exp_id in range(self.num_experts):
                mask = (expert_indices == exp_id)
                if mask.any():
                    tokens_for_expert = x_flat[mask]
                    exp_out = self.experts[exp_id](tokens_for_expert)
                    dynamic_out[mask] += weights[mask] * exp_out

        # Combine static anchor + dynamic MoE output
        combined_out = shared_out + dynamic_out
        combined_out = combined_out.view(batch_size, seq_len, d_model)

        # --- Auxiliary Load-Balancing Loss (Switch Transformer style) ---
        # Prevents expert collapse onto a small subset
        if self.training:
            # Fraction of tokens routed to each expert: f_i
            # One-hot representation of token assignments: (N_tokens, top_k, num_experts)
            one_hot_topk = F.one_hot(topk_indices, num_classes=self.num_experts).float()
            tokens_per_expert = one_hot_topk.sum(dim=(0, 1)) # (num_experts,)
            f_i = tokens_per_expert / (num_tokens * self.top_k)

            # Average router probability per expert: P_i
            p_i = router_probs.mean(dim=0) # (num_experts,)

            # Aux Loss = num_experts * sum(f_i * p_i) * coeff
            aux_loss = self.aux_loss_coeff * self.num_experts * torch.sum(f_i * p_i)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return combined_out, aux_loss


class NoolAlphaBlock(nn.Module):
    """
    Nool-Alpha Transformer Block (1 of 24).
    Pipeline:
      x -> Pre-RMSNorm1 -> GSLA Attention -> + Residual ->
      x -> Pre-RMSNorm2 -> HFK-MoE Block -> + Residual
    """
    def __init__(self, config: NoolAlphaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = GroupedSubspaceLatentAttention(config, layer_idx)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.moe = HeterogeneousFactorizedMoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 1. GSLA Attention with Pre-RMSNorm
        normed_x = self.norm1(hidden_states)
        attn_out, next_cache = self.attn(
            normed_x,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_out

        # 2. HFK-MoE with Pre-RMSNorm
        normed_x2 = self.norm2(hidden_states)
        moe_out, aux_loss = self.moe(normed_x2)
        hidden_states = hidden_states + moe_out

        return hidden_states, aux_loss, next_cache


class NoolAlphaForCausalLM(nn.Module):
    """
    Nool-Alpha-1.5B Causal Language Model.
    
    Complete architectural implementation:
      1. Token Embedding: W_embed in R^(vocab_size x d_model)
      2. 24 Transformer Blocks (GSLA + HFK-MoE)
      3. Final RMSNorm
      4. Global Residual Highway: x_final = x_24 + tanh(alpha) * RMSNorm(x_0)
      5. Tied-weight output projection (W_embed^T)
      6. Logit Soft-Capping: 30.0 * tanh(logits / 30.0)
    """
    def __init__(self, config: NoolAlphaConfig):
        super().__init__()
        self.config = config

        # Token Embedding (W_embed: tied weight with LM head)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Global Highway Early Normalization: RMSNorm(x_0)
        self.highway_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        # Learnable scalar parameter alpha, initialized at 0.05
        self.highway_alpha = nn.Parameter(torch.tensor(config.highway_alpha_init, dtype=torch.float32))

        # Transformer Blocks (Layer 1 through N_layers)
        self.layers = nn.ModuleList([
            NoolAlphaBlock(config, layer_idx) for layer_idx in range(config.n_layers)
        ])

        # Final RMSNorm (after last layer)
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # LM Head (Tied weights with embed_tokens)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Returns:
          logits: (batch, seq_len, vocab_size)
          loss: Cross-Entropy loss + MoE aux loss (if labels provided)
          total_aux_loss: aggregated MoE aux loss across all layers
          next_kv_caches: updated KV caches if use_cache is True
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Input embedding: x_0
        x_0 = self.embed_tokens(input_ids) # (batch, seq_len, d_model)
        
        # 2. Highway early norm representation: RMSNorm(x_0)
        x_0_norm = self.highway_norm(x_0)

        # 3. Forward pass through transformer layers
        hidden_states = x_0
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        next_caches = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_cache = past_key_values[idx] if past_key_values is not None else None
            if getattr(self.config, "gradient_checkpointing", False) and self.training and not use_cache:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, aux_loss, next_cache = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    layer_cache,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, aux_loss, next_cache = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    kv_cache=layer_cache,
                    use_cache=use_cache,
                )
            total_aux_loss = total_aux_loss + aux_loss
            if use_cache:
                next_caches.append(next_cache)

        # 4. Final layer normalization on x_24
        x_24_norm = self.final_norm(hidden_states)

        # 5. Global Residual Highway: x_final = x_24 + tanh(alpha) * RMSNorm(x_0)
        highway_factor = torch.tanh(self.highway_alpha)
        x_final = x_24_norm + highway_factor * x_0_norm

        # 6. Output Logits (Tied projection)
        logits_raw = self.lm_head(x_final)

        # 7. Logit Soft-Capping: 30.0 * tanh(logits_raw / 30.0)
        cap = self.config.logit_soft_cap
        if cap is not None and cap > 0:
            logits = cap * torch.tanh(logits_raw / cap)
        else:
            logits = logits_raw

        # 8. Compute Cross-Entropy Loss if labels are provided
        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            # Total loss combines LM Cross-Entropy + MoE Aux Load-Balancing Loss
            loss = ce_loss + total_aux_loss

        return logits, loss, total_aux_loss, next_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with temperature and top-p sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Pass inputs up to maximum position embeddings
            idx_cond = input_ids if input_ids.shape[1] <= self.config.max_position_embeddings else input_ids[:, -self.config.max_position_embeddings:]
            logits, _, _, _ = self(idx_cond)
            # Focus only on the last time step
            next_token_logits = logits[:, -1, :]

            if temperature <= 0.0:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                
                # Apply Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def get_num_params(self) -> Tuple[int, int]:
        """Returns (total_params, active_params)."""
        total = sum(p.numel() for p in self.parameters())
        # Active params: embeddings + all layers (attention + shared FFN + top_k/num_experts * dynamic experts) + norms + head
        # We can calculate active parameters analytically:
        shared_params = sum(p.numel() for n, p in self.named_parameters() if "experts" not in n)
        # Low rank experts params per expert
        expert_params = sum(p.numel() for p in self.layers[0].moe.experts[0].parameters()) * self.config.n_layers
        active = shared_params + (self.config.top_k_experts * expert_params)
        return total, active
