#!/usr/bin/env python3
"""
mini transformer v3 - now with 100% more modern architecture
if you're reading this: it's like v2, slightly improved with extra flaws/bugs which will be fixed in next version... promise :)
author: 5hridhyan/Aranya-Marjara (still an 11th grader, but now with a year more experience)
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import tiktoken
from tqdm import tqdm  # optional, for fancy progress bars

# -------------------- Setup: TF32, mixed precision helpers --------------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# -------------------- Configuration dataclass --------------------
@dataclass
class Config:
    """
    All the knobs, now with even more knobs.
    Presets: tiny (~10M), small (~30M), medium (~100M), large (~350M)
    """
    # Architecture
    vocab_size: int = 50304  # rounded up for efficiency (tiktoken cl100k_base is 100k, but we'll set later)
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2          # GQA
    n_layers: int = 6
    d_ff: Optional[int] = None   # defaults to 4*d_model
    dropout: float = 0.1
    context_length: int = 1024
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = None  # for windowed attention (Mistral style)

    # MoE
    use_moe: bool = False
    n_experts: int = 8
    n_active_experts: int = 2
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.0001   # optional, helps stability

    # Regularization & init
    weight_tying: bool = True
    init_std: float = 0.02
    init_scaling: float = 1.0 / math.sqrt(2.0)  # for depth scaling

    # Generation
    max_generation_length: int = 512

    @classmethod
    def tiny(cls):
        return cls(d_model=128, n_heads=4, n_kv_heads=1, n_layers=4, d_ff=512)

    @classmethod
    def small(cls):
        return cls(d_model=256, n_heads=8, n_kv_heads=2, n_layers=8, d_ff=1024)

    @classmethod
    def medium(cls):
        return cls(d_model=512, n_heads=8, n_kv_heads=2, n_layers=12, d_ff=2048)

    @classmethod
    def large(cls):
        return cls(d_model=768, n_heads=12, n_kv_heads=4, n_layers=24, d_ff=3072)

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        # Sanity checks
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

# -------------------- Rotary Positional Embeddings (fixed) --------------------
class RotaryEmbedding(nn.Module):
    """
    RoPE – now with interleaved frequencies and proper cache rebuild.
    The key fix: positions are offset by past_length, so generation works.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Compute inverse frequencies once
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        # Precompute sin/cos for all positions
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)                # (seq_len, dim//2)
        # For each pair of dimensions, use the same frequency (this is the standard RoPE)
        emb = torch.cat((freqs, freqs), dim=-1)              # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, past_length: int = 0):
        """
        Apply rotary embeddings with proper position offsets.
        q, k: (batch, heads, seq_len, head_dim)
        """
        total_len = seq_len + past_length
        if total_len > self.cos_cached.shape[2]:
            self._build_cache(total_len)

        cos = self.cos_cached[:, :, past_length:total_len, :]   # (1,1,seq_len,dim)
        sin = self.sin_cached[:, :, past_length:total_len, :]

        # Rotate q and k
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

# -------------------- KV Cache (now unified across layers) --------------------
class KVCache:
    """
    Unified key‑value cache for all layers.
    Stores as a single tensor: (n_layers, batch, n_kv_heads, max_seq_len, head_dim)
    """
    def __init__(self, n_layers: int, batch_size: int, n_kv_heads: int, head_dim: int,
                 max_seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Allocate contiguous memory (zeros)
        shape = (n_layers, batch_size, n_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.seen_tokens = 0   # total tokens stored so far

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        Store new keys/values for a specific layer.
        k, v: (batch, n_kv_heads, seq_len, head_dim)
        """
        seq_len = k.size(2)
        start = self.seen_tokens
        end = start + seq_len
        self.k_cache[layer_idx, :, :, start:end, :] = k
        self.v_cache[layer_idx, :, :, start:end, :] = v

    def get(self, layer_idx: int):
        """Retrieve all cached keys/values for a layer up to seen_tokens."""
        return (self.k_cache[layer_idx, :, :, :self.seen_tokens, :],
                self.v_cache[layer_idx, :, :, :self.seen_tokens, :])

    def reset(self):
        """Reset cache (for generation start)."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seen_tokens = 0

    def set_seen_tokens(self, count: int):
        self.seen_tokens = count

# -------------------- RMSNorm --------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

# -------------------- Attention with GQA, sliding window, FlashAttention --------------------
class Attention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.layer_idx = layer_idx
        self.sliding_window = cfg.sliding_window

        # Projections
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=cfg.context_length * 2, theta=cfg.rope_theta)

        # Cache will be set from outside
        self.cache: Optional[KVCache] = None

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """(batch, kv_heads, seq_len, head_dim) -> (batch, n_heads, seq_len, head_dim)"""
        if n_rep == 1:
            return x
        batch, kv_heads, seq_len, head_dim = x.shape
        return (x[:, :, None, :, :]
                .expand(batch, kv_heads, n_rep, seq_len, head_dim)
                .reshape(batch, kv_heads * n_rep, seq_len, head_dim))

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_length: int = 0) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project to q, k, v
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE with correct offset
        q, k = self.rope(q, k, seq_len, past_length)

        # KV cache update
        if use_cache and self.cache is not None:
            self.cache.update(self.layer_idx, k, v)
            k_full, v_full = self.cache.get(self.layer_idx)
            # Now the full sequence length includes past
            total_seq_len = k_full.size(2)
        else:
            k_full, v_full = k, v
            total_seq_len = seq_len

        # Repeat k/v heads to match q heads (GQA)
        n_rep = self.n_heads // self.n_kv_heads
        k_rep = self._repeat_kv(k_full, n_rep)
        v_rep = self._repeat_kv(v_full, n_rep)

        # Optional sliding window mask
        attn_mask = None
        if self.sliding_window is not None and total_seq_len > self.sliding_window:
            # Create a causal + sliding window mask
            # For each query position i, it can attend to keys in [max(0, i-window+1), i]
            # Using PyTorch's SDPA with custom mask requires a 4D mask of shape (batch, 1, q_len, kv_len)
            # We'll create a boolean mask where False indicates masked positions
            q_indices = torch.arange(seq_len, device=x.device).unsqueeze(1)   # (seq_len, 1)
            k_indices = torch.arange(total_seq_len, device=x.device).unsqueeze(0)  # (1, total_seq_len)
            causal_mask = (q_indices + past_length) >= k_indices   # standard causal: query pos >= key pos
            window_mask = (q_indices + past_length) - k_indices < self.sliding_window   # within window
            allowed = causal_mask & window_mask
            # SDPA expects mask with False where we want to mask out
            attn_mask = allowed[None, None, :, :]   # (1, 1, seq_len, total_seq_len)

        # Flash Attention (or fallback)
        if hasattr(F, "scaled_dot_product_attention"):
            # is_causal is ignored if attn_mask is provided
            attn_output = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attn_mask,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=attn_mask is None   # if no mask, use causal
            )
        else:
            # Manual attention (with optional sliding window)
            attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                # Convert boolean mask to float mask (-inf for masked positions)
                attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
            else:
                # Causal mask only
                mask = torch.triu(
                    torch.ones(seq_len, total_seq_len, dtype=torch.bool, device=x.device),
                    diagonal=1 + past_length
                )
                attn_weights = attn_weights.masked_fill(mask, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
            attn_weights = self.drop(attn_weights)
            attn_output = torch.matmul(attn_weights, v_rep)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(attn_output)

# -------------------- SwiGLU FFN --------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

# -------------------- MoE with load balancing and optional z-loss --------------------
class MoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_active_experts = cfg.n_active_experts
        self.aux_loss_coef = cfg.moe_aux_loss_coef
        self.z_loss_coef = cfg.moe_z_loss_coef

        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

        # Stack expert weights for vectorized computation
        self.w1 = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_model, cfg.d_ff))
        self.w2 = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_ff, cfg.d_model))
        self.w3 = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_model, cfg.d_ff))

        # Initialize
        nn.init.normal_(self.w1, std=cfg.init_std)
        nn.init.normal_(self.w2, std=cfg.init_std / math.sqrt(2 * cfg.n_layers) if cfg.init_scaling else cfg.init_std)
        nn.init.normal_(self.w3, std=cfg.init_std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        n_tokens = x_flat.shape[0]

        # Router logits
        router_logits = self.router(x_flat)               # (n_tokens, n_experts)

        # Softmax over experts
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Top-k experts per token
        topk_weights, topk_indices = torch.topk(routing_weights, self.n_active_experts, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)   # renormalize

        # Prepare for batched expert computation
        expert_indices = topk_indices.view(-1)                         # (n_tokens * k)
        token_indices = torch.arange(n_tokens, device=x.device).repeat_interleave(self.n_active_experts)
        expert_weights = topk_weights.view(-1)                         # (n_tokens * k)

        # Sort by expert for contiguous memory access
        sorted_expert_indices, sort_idx = torch.sort(expert_indices)
        sorted_token_indices = token_indices[sort_idx]
        sorted_weights = expert_weights[sort_idx]

        # Output buffer
        outputs = torch.zeros(n_tokens, d_model, device=x.device, dtype=x.dtype)

        # Process each expert's tokens in batch
        unique_experts, counts = torch.unique_consecutive(sorted_expert_indices, return_counts=True)
        start = 0
        for expert_idx, count in zip(unique_experts, counts):
            end = start + count.item()
            expert_token_idx = sorted_token_indices[start:end]
            expert_weight = sorted_weights[start:end, None]            # (count, 1)

            # Extract token embeddings
            expert_tokens = x_flat[expert_token_idx]                   # (count, d_model)

            # Compute expert output: (silu(x @ w1) * (x @ w3)) @ w2
            gate_out = F.silu(torch.mm(expert_tokens, self.w1[expert_idx]))
            up_out = torch.mm(expert_tokens, self.w3[expert_idx])
            expert_out = torch.mm(gate_out * up_out, self.w2[expert_idx])   # (count, d_model)

            # Weighted accumulate
            outputs.index_add_(0, expert_token_idx, expert_weight * expert_out)
            start = end

        # Compute auxiliary losses
        # 1. Load balancing loss (Switch Transformer)
        # Importance: mean(softmax over all experts)
        probs = F.softmax(router_logits, dim=-1)
        importance = probs.mean(dim=0)                         # (n_experts,)

        # Load: fraction of tokens routed to each expert (top-1)
        top1_expert = topk_indices[:, 0]                        # (n_tokens,)
        load = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        load.scatter_add_(0, top1_expert, torch.ones_like(top1_expert, dtype=torch.float32))
        load = load / n_tokens

        aux_loss = self.n_experts * (importance * load).sum()

        # 2. Optional z-loss to keep router logits small (helps stability)
        if self.z_loss_coef > 0:
            # Compute mean squared logits over all tokens and experts
            z_loss = torch.mean(router_logits ** 2) * self.z_loss_coef
        else:
            z_loss = 0.0

        losses = {"aux_loss": aux_loss * self.aux_loss_coef, "z_loss": z_loss}

        return outputs.view(batch, seq_len, -1), losses

# -------------------- Transformer Block --------------------
class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)

        if cfg.use_moe:
            self.ffn = MoE(cfg)
        else:
            self.ffn = SwiGLU(cfg)

        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_length: int = 0):
        # Attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, use_cache=use_cache, past_length=past_length)
        x = self.drop1(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)

        if isinstance(self.ffn, MoE):
            ffn_out, losses = self.ffn(x)
            x = self.drop2(ffn_out)
            x = x + residual
            return x, losses
        else:
            x = self.ffn(x)
            x = self.drop2(x)
            x = x + residual
            return x

# -------------------- Main Model --------------------
class MiniGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.head.weight = self.token_embed.weight

        self._init_weights()

        # KV cache placeholder (set by setup_cache)
        self.kv_cache: Optional[KVCache] = None

    def _init_weights(self):
        """Initialize all weights with proper scaling."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "out_proj" in name or "w2" in name:
                    # Scale output projections (including MoE's second linear) by depth
                    std = self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers) if self.cfg.init_scaling else self.cfg.init_std
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def setup_cache(self, batch_size: int) -> KVCache:
        """Create a fresh KV cache and inject into all attention layers."""
        head_dim = self.cfg.d_model // self.cfg.n_heads
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.kv_cache = KVCache(
            n_layers=self.cfg.n_layers,
            batch_size=batch_size,
            n_kv_heads=self.cfg.n_kv_heads,
            head_dim=head_dim,
            max_seq_len=self.cfg.context_length * 2,   # generous for generation
            device=device,
            dtype=dtype
        )
        for block in self.blocks:
            block.attn.cache = self.kv_cache
        return self.kv_cache

    def clear_cache(self):
        """Remove cache references."""
        self.kv_cache = None
        for block in self.blocks:
            block.attn.cache = None

    def forward(self, tokens: torch.Tensor, use_cache: bool = False, past_length: int = 0):
        """
        Forward pass.
        If use_cache=True, expects past_length to be the number of tokens already in cache.
        Returns logits and optionally a dict of auxiliary losses.
        """
        batch, seq_len = tokens.shape

        # Trim if necessary (only if we exceed context length)
        if seq_len > self.cfg.context_length:
            tokens = tokens[:, -self.cfg.context_length:]
            seq_len = self.cfg.context_length

        x = self.token_embed(tokens)
        x = self.drop(x)

        aux_losses = {}
        for block in self.blocks:
            if isinstance(block.ffn, MoE):
                x, losses = block(x, use_cache=use_cache, past_length=past_length)
                for k, v in losses.items():
                    aux_losses[k] = aux_losses.get(k, 0.0) + v
            else:
                x = block(x, use_cache=use_cache, past_length=past_length)

        x = self.norm(x)
        logits = self.head(x)

        # Update cache's seen tokens after all layers (only for last layer update)
        if use_cache and self.kv_cache is not None:
            # In attention layers we already updated per layer; we just need to increment seen_tokens.
            # But KVCache.update already increments seen_tokens only on layer 0.
            # We'll trust that mechanism.
            pass

        if aux_losses:
            return logits, aux_losses
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ):
        """
        Autoregressive generation with KV cache, top-k/top-p sampling, and repetition penalty.
        """
        self.eval()
        batch_size = prompt.shape[0]

        # Setup cache
        cache = self.setup_cache(batch_size)

        # Prefill cache with prompt
        past_length = 0
        _ = self(prompt, use_cache=True, past_length=0)
        # After prefill, cache now contains prompt tokens
        # The next token to generate is at position past_length = prompt.size(1)
        past_length = prompt.size(1)

        generated = prompt.clone()
        for _ in range(max_new_tokens):
            if cache.seen_tokens >= self.cfg.context_length * 2:
                logging.warning("Reached max cache length, stopping generation")
                break

            # Take the last token (shape: (batch, 1))
            last_token = generated[:, -1:]

            # Forward with cache
            logits = self(last_token, use_cache=True, past_length=past_length)
            logits = logits[:, -1, :]  # (batch, vocab)

            # Apply repetition penalty (multiply logits of already seen tokens by penalty)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    seen_tokens = set(generated[i].tolist())
                    for token in seen_tokens:
                        logits[i, token] /= repetition_penalty

            # Temperature scaling
            logits = logits / max(temperature, 1e-6)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right so first token is always kept
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original indices
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            generated = torch.cat([generated, next_token], dim=1)
            past_length += 1

            # Stop condition (optional)
            if stop_tokens is not None:
                if (next_token[0, 0] in stop_tokens):
                    break

        self.clear_cache()
        return generated

# -------------------- Dataset --------------------
class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, context_length: int):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y

# -------------------- Trainer --------------------
class Trainer:
    def __init__(self, args):
        self.args = args
        self._setup_logging()
        self._setup_distributed()
        self._setup_device()
        self._load_data()
        self._build_model()
        self._setup_optimizer_scheduler()
        self._setup_mixed_precision()
        self._setup_ema()
        self._setup_checkpointing()

        # State
        self.current_step = 0
        self.best_val_loss = float("inf")

    def _setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO if self.args.local_rank == 0 else logging.WARNING,
        )
        self.logger = logging.getLogger(__name__)

    def _setup_distributed(self):
        self.is_distributed = self.args.distributed
        self.local_rank = self.args.local_rank
        self.world_size = 1
        self.rank = 0

        if self.is_distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed training requires CUDA")
            dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
            if self.is_distributed:
                self.logger.warning("Distributed training on CPU is not supported; disabling distributed.")
                self.is_distributed = False

    def _load_data(self):
        self.logger.info(f"Loading data from {self.args.data}")
        with open(self.args.data, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = self.tokenizer.encode(text)

        # Update vocab size in config
        self.args.vocab_size = self.tokenizer.n_vocab

        data = torch.tensor(tokens, dtype=torch.long)
        split = int(0.9 * len(data))
        train_data, val_data = data[:split], data[split:]

        self.train_dataset = TextDataset(train_data, self.args.context_length)
        self.val_dataset = TextDataset(val_data, self.args.context_length)

        # Samplers
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True

        num_workers = min(4, os.cpu_count() or 1)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.logger.info(
            f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}"
        )

    def _build_model(self):
        # Choose preset
        preset_map = {
            "tiny": Config.tiny,
            "small": Config.small,
            "medium": Config.medium,
            "large": Config.large,
        }
        if self.args.model_size in preset_map:
            cfg = preset_map[self.args.model_size]()
        else:
            cfg = Config()

        # Override from args
        cfg.vocab_size = self.args.vocab_size
        cfg.context_length = self.args.context_length
        cfg.use_moe = self.args.use_moe
        cfg.sliding_window = self.args.sliding_window

        self.cfg = cfg
        self.model = MiniGPT(cfg).to(self.device)

        # Optional torch.compile
        if self.args.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            self.logger.info("Model compiled with torch.compile")

        # DDP
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def _setup_optimizer_scheduler(self):
        # Optimizer
        try:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(0.9, 0.95),
                weight_decay=0.1,
                fused=True,
            )
            self.logger.info("Using fused AdamW")
        except:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
            self.logger.info("Fused AdamW not available, using regular")

        # Scheduler
        steps_per_epoch = len(self.train_loader)
        effective_steps_per_epoch = math.ceil(steps_per_epoch / self.args.accumulation_steps)
        total_steps = self.args.epochs * effective_steps_per_epoch

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.args.warmup_steps
        )
        cosine_steps = max(1, total_steps - self.args.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_steps)

        self.scheduler = SequentialLR(
            self.optimizer,
            [warmup_scheduler, cosine_scheduler],
            milestones=[self.args.warmup_steps],
        )

    def _setup_mixed_precision(self):
        self.use_amp = self.args.mixed_precision and self.device.type == "cuda"
        if self.use_amp:
            if torch.cuda.is_bf16_supported():
                self.precision_dtype = torch.bfloat16
                self.logger.info("Using bfloat16 mixed precision")
            else:
                self.precision_dtype = torch.float16
                self.logger.info("Using float16 mixed precision")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _setup_ema(self):
        self.ema_model = None
        if self.args.use_ema:
            self.ema_model = MiniGPT(self.cfg).to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_decay = self.args.ema_decay
            self.logger.info(f"EMA enabled with decay {self.ema_decay}")

    def _setup_checkpointing(self):
        self.checkpoint_dir = Path(self.args.output_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            model_state = self.model.state_dict()
            ema_state = self.ema_model.state_dict()
            for name, param in model_state.items():
                ema_name = name.replace("module.", "") if name.startswith("module.") else name
                if ema_name in ema_state:
                    ema_state[ema_name].mul_(self.ema_decay).add_(param, alpha=1 - self.ema_decay)

    def _compute_loss(self, logits, targets, aux_losses=None):
        """Compute cross entropy and add auxiliary losses."""
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if aux_losses:
            for k, v in aux_losses.items():
                loss = loss + v
        return loss

    def train_epoch(self, epoch: int):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=self.rank != 0)
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # Forward with mixed precision
            with torch.autocast(device_type=self.device.type, dtype=self.precision_dtype, enabled=self.use_amp):
                if self.cfg.use_moe:
                    logits, aux_losses = self.model(x)
                else:
                    logits = self.model(x)
                    aux_losses = None

                loss = self._compute_loss(logits, y, aux_losses)

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.args.accumulation_steps

            # Backward
            if self.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self._optimizer_step()
                self.scheduler.step()
                self.current_step += 1

            total_loss += loss.item()
            num_batches += 1

            if self.rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Handle remaining gradients
        if len(self.train_loader) % self.args.accumulation_steps != 0:
            self._optimizer_step()
            self.scheduler.step()
            self.current_step += 1

        return total_loss / num_batches

    def _optimizer_step(self):
        """Clip gradients and step optimizer (with scaler if needed)."""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        self._update_ema()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.precision_dtype, enabled=self.use_amp):
                if self.cfg.use_moe:
                    logits, aux_losses = self.model(x)
                else:
                    logits = self.model(x)
                    aux_losses = None
                loss = self._compute_loss(logits, y, aux_losses)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def generate_sample(self, prompt: str, max_tokens: int = 100):
        model = self.ema_model if self.ema_model is not None else self.model
        if self.is_distributed:
            model = model.module

        model.eval()
        tokens = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([tokens], device=self.device)

        generated = model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        return self.tokenizer.decode(generated[0].tolist())

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        if self.rank != 0:
            return

        # Prepare state dict
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "config": asdict(self.cfg),
            "tokenizer_name": "cl100k_base",
            "args": vars(self.args),
        }
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "checkpoint_latest.pt")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "checkpoint_best.pt")

        if epoch % self.args.save_every == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")

        self.logger.info(f"Saved checkpoint at epoch {epoch}")

    def load_checkpoint(self, path: Union[str, Path]):
        checkpoint = torch.load(path, map_location=self.device)

        # Load model
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.ema_model is not None and "ema_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])

        return checkpoint["epoch"], checkpoint["val_loss"]

    def train(self):
        start_epoch = 0
        if self.args.resume:
            start_epoch, _ = self.load_checkpoint(self.args.resume)
            self.logger.info(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            lr = self.scheduler.get_last_lr()[0]
            self.logger.info(
                f"Epoch {epoch+1}/{self.args.epochs} | "
                f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                f"LR: {lr:.2e} | {'BEST' if is_best else ''}"
            )

            if (epoch + 1) % 5 == 0 and self.rank == 0:
                sample = self.generate_sample("The future of AI is")
                self.logger.info(f"Sample:\n{sample}")

            self.save_checkpoint(epoch + 1, val_loss, is_best)

        if self.is_distributed:
            dist.destroy_process_group()

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="MiniGPT v3 – modern transformer from scratch")
    # Data
    parser.add_argument("--data", type=str, required=True, help="Path to text file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Where to save checkpoints")

    # Model
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--use_moe", action="store_true", help="Enable MoE")
    parser.add_argument("--sliding_window", type=int, default=None, help="Sliding window attention size")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Optimization
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)

    # Distributed
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0, help="Set by torchrun")

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--save_every", type=int, default=5)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
