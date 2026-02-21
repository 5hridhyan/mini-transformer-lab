#!/usr/bin/env python3
"""
mini transformer v2
if you're reading this: yes, it actually works. no, i don't know why either.
author: 5hridhyan/Aranya-Marjara (yeah just a 11th grader if you can make it better do it, since I couldn't!)
"""

import argparse
import math
import time
import json
import pickle
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import tiktoken

# tf32 go brrr (free speed on ampere+)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


@dataclass
class Config:
    """
    all the knobs you can turn
    default values work for small experiments on igpu
    
    presets:
        tiny: ~10m params (runs on anything)
        small: ~30m params (needs a half-decent gpu)
        medium: ~100m params (for when you borrowed someone's 4090)
    """
    # architecture (the important bits)
    vocab_size: int = 5000
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 2  # gqa ratio = n_heads // n_kv_heads
    n_layers: int = 4
    d_ff: Optional[int] = None  # will be 4*d_model if you're lazy
    dropout: float = 0.1
    context_length: int = 256
    rope_theta: float = 10000.0  # the classic
    
    # moe (off by default because it's expensive)
    use_moe: bool = False
    n_experts: int = 8
    n_active_experts: int = 2
    moe_aux_loss_coef: float = 0.01
    
    # training misc
    weight_tying: bool = True
    max_batch_size: int = 32
    
    @classmethod
    def tiny(cls):
        """for when you're on integrated graphics (vega 8 gang rise up)"""
        return cls(d_model=128, n_heads=4, n_kv_heads=1, n_layers=3, d_ff=512)
    
    @classmethod
    def small(cls):
        """for when you have a half-decent gpu"""
        return cls(d_model=256, n_heads=8, n_kv_heads=2, n_layers=6, d_ff=1024)
    
    @classmethod
    def medium(cls):
        """for when you borrowed someone's 4090 and want to feel something"""
        return cls(d_model=512, n_heads=8, n_kv_heads=2, n_layers=12, d_ff=2048)
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


class KVTheSaverOfTimeAndSanity:
    """
    this little guy remembers everything so you don't have to recompute it
    like that one friend who remembers every embarrassing thing you've ever done
    
    now allocates directly on device because cpu->gpu transfers are for chumps
    """
    def __init__(self, max_batch: int, max_seq_len: int, n_layers: int, n_kv_heads: int, head_dim: int, device: torch.device):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self._current_length = 0
        
        # allocate directly on target device (we're not animals)
        self.k_cache = []
        self.v_cache = []
        for _ in range(n_layers):
            self.k_cache.append(torch.zeros(max_batch, n_kv_heads, max_seq_len, head_dim, device=device))
            self.v_cache.append(torch.zeros(max_batch, n_kv_heads, max_seq_len, head_dim, device=device))
    
    @property
    def current_length(self) -> int:
        return self._current_length
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """store new memories at the current position"""
        batch, heads, seq_len, dim = k.shape
        start_pos = self._current_length
        end_pos = start_pos + seq_len
        
        self.k_cache[layer_idx][:, :, start_pos:end_pos] = k
        self.v_cache[layer_idx][:, :, start_pos:end_pos] = v
        
        # update global length after last layer (so everyone agrees what time it is)
        if layer_idx == self.n_layers - 1:
            self._current_length = end_pos
    
    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """gimme everything you've got (up to current length)"""
        return (self.k_cache[layer_idx][:, :, :self._current_length, :],
                self.v_cache[layer_idx][:, :, :self._current_length, :])
    
    def reset(self):
        """selective amnesia (for generation only)"""
        self._current_length = 0
        for i in range(self.n_layers):
            self.k_cache[i].zero_()
            self.v_cache[i].zero_()


class SpinnyBoi(nn.Module):
    """
    rope - because adding position vectors is so 2017
    now with correct position offset because i'm not an animal
    
    this rotates queries and keys in complex space
    yes it's magic. no i don't fully understand it either.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # this formula was stolen from the original rope paper
        # don't tell them
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """precompute cos and sin for all positions (trig is expensive)"""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, past_length: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        apply rotary embeddings with correct position offset
        past_length: how many tokens came before (for generation)
        
        this is the fixed version that actually works
        previous version was generating gibberish past the first token
        """
        total_len = seq_len + past_length
        if total_len > self.cos_cached.shape[2]:
            self._build_cache(total_len)
        
        # get the right slice of cos/sin for current positions
        # this is the key fix: positions start at past_length, not 0
        cos = self.cos_cached[:, :, past_length:total_len, :]
        sin = self.sin_cached[:, :, past_length:total_len, :]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """the secret sauce - rotate half the dimensions"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class MakeNumbersNormalAgain(nn.Module):
    """
    rmsnorm - like layernorm but without the mean
    because who needs averages anyway
    
    faster and works just as well (sometimes better)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class LookAtMeImTheAttentionNow(nn.Module):
    """
    gqa - because mha uses too much vram
    also has flash attention because we're not savages
    
    this version:
    - stores original kv heads (not repeated ones)
    - applies rope with correct offsets
    - uses flash attention when available
    - doesn't waste memory on repeated cache
    """
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.layer_idx = layer_idx
        
        # projections (bias=false because we're efficient)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = SpinnyBoi(self.head_dim, max_seq_len=cfg.context_length * 2, theta=cfg.rope_theta)
        
        # cache will be injected from parent (dependency injection baby)
        self.cache: Optional[KVTheSaverOfTimeAndSanity] = None
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """make fewer heads into more heads (it's broadcasting, not magic)"""
        if n_rep == 1:
            return x
        batch, n_kv_heads, seq_len, head_dim = x.shape
        return x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim).reshape(
            batch, n_kv_heads * n_rep, seq_len, head_dim
        )
    
    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # project to q, k, v (the holy trinity)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # get past length from cache (if any)
        past_length = 0
        if use_cache and self.cache is not None:
            past_length = self.cache.current_length
        
        # apply rope with correct position offset
        # this is the fix: new tokens get positions = past_length + their index
        q, k = self.rope(q, k, seq_len, past_length)
        
        # handle kv cache (store ORIGINAL kv heads, not repeated ones)
        if use_cache and self.cache is not None:
            # store original kv (before repeating)
            self.cache.update(self.layer_idx, k, v)
            
            # get full kv including past
            k_full, v_full = self.cache.get(self.layer_idx)
        else:
            k_full, v_full = k, v
        
        # repeat kv for gqa (only for attention computation)
        n_rep = self.n_heads // self.n_kv_heads
        k_rep = self._repeat_kv(k_full, n_rep)
        v_rep = self._repeat_kv(v_full, n_rep)
        
        # flash attention go brrr (if available)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=None,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=True  # always causal, even with cache
            )
        else:
            # fallback for old pytorch (like my laptop)
            attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale
            
            # proper causal mask with past length
            mask = torch.triu(
                torch.ones(seq_len, k_rep.shape[2], dtype=torch.bool, device=x.device),
                diagonal=1 + past_length
            )
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
            attn_weights = self.drop(attn_weights)
            attn_output = torch.matmul(attn_weights, v_rep)
        
        # reshape and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(attn_output)


class SwiggleGLU(nn.Module):
    """
    swiglu - when gelu isn't fancy enough for you
    used in palm, llama, and every other model that wants to sound impressive
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
    
    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class SpaghettiOfExperts(nn.Module):
    """
    moe - because one ffn isn't enough, we need eight
    now with correct switch transformer load balancing loss
    
    each token gets routed to top-k experts
    the router learns to distribute tokens evenly (or else)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_active_experts = cfg.n_active_experts
        self.aux_loss_coef = cfg.moe_aux_loss_coef
        
        # router (decides who gets what)
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        
        # expert weights (stacked for vectorized computation)
        # this is way faster than having separate modules
        self.w1 = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_model, cfg.d_ff))
        self.w2 = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_ff, cfg.d_model))
        self.w3 = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_model, cfg.d_ff))
        
        # init with the usual magic number
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        nn.init.normal_(self.w3, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        n_tokens = x_flat.shape[0]
        
        # router logits (who wants this token?)
        router_logits = self.router(x_flat)
        
        # get top-k experts (the popular kids)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.n_active_experts, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # prepare for batched computation
        expert_indices = topk_indices.view(-1)
        token_indices = torch.arange(n_tokens, device=x.device).repeat_interleave(self.n_active_experts)
        expert_weights = topk_weights.view(-1)
        
        # sort by expert for contiguous access (cache efficiency ftw)
        sorted_expert_indices, sort_idx = torch.sort(expert_indices)
        sorted_token_indices = token_indices[sort_idx]
        sorted_weights = expert_weights[sort_idx]
        
        # find expert boundaries (where each expert's tokens start/end)
        unique_experts, counts = torch.unique_consecutive(sorted_expert_indices, return_counts=True)
        
        # output tensor
        outputs = torch.zeros(n_tokens, d_model, device=x.device)
        
        # process each expert's batch (no loops over tokens, only experts)
        start_idx = 0
        for expert_idx, count in zip(unique_experts, counts):
            if count == 0:
                continue
            
            end_idx = start_idx + count.item()
            
            # get tokens for this expert
            expert_token_indices = sorted_token_indices[start_idx:end_idx]
            expert_weights_batch = sorted_weights[start_idx:end_idx, None]
            
            # extract tokens
            expert_tokens = x_flat[expert_token_indices]
            
            # compute expert output: silu(w1*x) * (w3*x) then w2
            gate_out = F.silu(torch.mm(expert_tokens, self.w1[expert_idx]))
            up_out = torch.mm(expert_tokens, self.w3[expert_idx])
            expert_out = torch.mm(gate_out * up_out, self.w2[expert_idx])
            
            # weight and accumulate (index_add_ is magic)
            outputs.index_add_(0, expert_token_indices, expert_weights_batch * expert_out)
            
            start_idx = end_idx
        
        # correct switch transformer load balancing loss
        # loss = n_experts * sum(importance * load)
        # importance = mean(softmax(router_logits))
        # load = fraction of tokens routed to each expert (top-1 only, per switch paper)
        router_probs = F.softmax(router_logits, dim=-1)
        importance = router_probs.mean(dim=0)
        
        # compute load: for each expert, fraction of tokens that chose it as top-1
        top1_indices = topk_indices[:, 0]  # top-1 expert per token
        load = torch.zeros(self.n_experts, device=x.device)
        load.scatter_add_(0, top1_indices, torch.ones_like(top1_indices, dtype=torch.float32))
        load = load / n_tokens
        
        aux_loss = self.n_experts * (importance * load).sum()
        
        return outputs.view(batch, seq_len, -1), aux_loss


class BlockParty(nn.Module):
    """
    one transformer block, served fresh daily
    contains: attention, ffn, norms, residuals (the whole package)
    """
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.norm1 = MakeNumbersNormalAgain(cfg.d_model)
        self.norm2 = MakeNumbersNormalAgain(cfg.d_model)
        self.attn = LookAtMeImTheAttentionNow(cfg, layer_idx)
        
        if cfg.use_moe:
            self.ffn = SpaghettiOfExperts(cfg)
        else:
            self.ffn = SwiggleGLU(cfg)
        
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)
    
    def forward(self, x: torch.Tensor, use_cache: bool = False):
        # attention with residual (the classic)
        residual = x
        x = self.norm1(x)
        x = self.attn(x, use_cache=use_cache)
        x = self.drop1(x)
        x = x + residual
        
        # ffn with residual (also classic)
        residual = x
        x = self.norm2(x)
        
        if isinstance(self.ffn, SpaghettiOfExperts):
            x, aux_loss = self.ffn(x)
            x = self.drop2(x)
            x = x + residual
            return x, aux_loss
        else:
            x = self.ffn(x)
            x = self.drop2(x)
            x = x + residual
            return x


class MiniGPT(nn.Module):
    """
    the main event: a transformer that actually works now
    
    features:
    - rope (fixed)
    - gqa (working)
    - moe (optional)
    - weight tying (saves params)
    - proper initialization (doesn't explode)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # embeddings (tokens go brrr)
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        
        # transformer blocks (stack 'em high)
        self.blocks = nn.ModuleList([
            BlockParty(cfg, i) for i in range(cfg.n_layers)
        ])
        
        # final norm
        self.norm = MakeNumbersNormalAgain(cfg.d_model)
        
        # output head (back to tokens)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.weight_tying:
            self.head.weight = self.token_embed.weight  # the classic trick
        
        # init with some thought (depth scaling)
        self._init_weights()
        
        # generation cache (starts empty)
        self.kv_cache: Optional[KVTheSaverOfTimeAndSanity] = None
    
    def _init_weights(self):
        """because default init is for amateurs"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "out_proj" in name or "w2" in name:
                    # scale output projections by depth (thanks t5)
                    nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def setup_cache(self, batch_size: int) -> KVTheSaverOfTimeAndSanity:
        """create and inject cache into all attention layers"""
        head_dim = self.cfg.d_model // self.cfg.n_heads
        device = next(self.parameters()).device
        self.kv_cache = KVTheSaverOfTimeAndSanity(
            max_batch=batch_size,
            max_seq_len=self.cfg.context_length * 2,
            n_layers=self.cfg.n_layers,
            n_kv_heads=self.cfg.n_kv_heads,
            head_dim=head_dim,
            device=device
        )
        
        # inject into all attention layers
        for block in self.blocks:
            block.attn.cache = self.kv_cache
        
        return self.kv_cache
    
    def clear_cache(self):
        """remove cache references (cleanup after generation)"""
        self.kv_cache = None
        for block in self.blocks:
            block.attn.cache = None
    
    def forward(self, tokens: torch.Tensor, use_cache: bool = False):
        batch, seq_len = tokens.shape
        
        # trim if too long (like this comment)
        if seq_len > self.cfg.context_length:
            tokens = tokens[:, -self.cfg.context_length:]
            seq_len = self.cfg.context_length
        
        # embeddings
        x = self.token_embed(tokens)
        x = self.drop(x)
        
        # through blocks
        aux_losses = []
        for block in self.blocks:
            if self.cfg.use_moe:
                x, aux_loss = block(x, use_cache=use_cache)
                aux_losses.append(aux_loss)
            else:
                x = block(x, use_cache=use_cache)
        
        x = self.norm(x)
        logits = self.head(x)
        
        if aux_losses:
            return logits, sum(aux_losses) * self.cfg.moe_aux_loss_coef
        return logits
    
    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9):
        """
        generate text autoregressively
        now with:
        - working rope offsets
        - cache that doesn't explode
        - safety check for cache overflow
        """
        self.eval()
        
        # setup cache
        cache = self.setup_cache(prompt.shape[0])
        
        # prefill cache with prompt
        _ = self(prompt, use_cache=True)
        
        generated = prompt.clone()
        
        for _ in range(max_new_tokens):
            # safety check: don't exceed cache size
            if cache.current_length >= self.cfg.context_length * 2:
                print(f"warning: reached max cache length ({self.cfg.context_length * 2}), stopping generation")
                break
            
            # only last token (cache has the rest)
            last_token = generated[:, -1:]
            
            # forward with cache
            logits = self(last_token, use_cache=True)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            # top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # top-p filtering (nucleus)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # sample from filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        
        # clean up (good hygiene)
        self.clear_cache()
        
        return generated


class TextDataset(Dataset):
    """
    gives x and y where y = x shifted by 1
    it's not rocket science but someone had to write it
    """
    def __init__(self, data: torch.Tensor, context_length: int):
        self.data = data
        self.context_length = context_length
    
    def __len__(self):
        return max(0, len(self.data) - self.context_length)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y


class Trainer:
    """
    trainer object with all the modern fixings
    
    features:
    - distributed training (ddp)
    - mixed precision (bf16/fp16)
    - gradient accumulation
    - ema (exponential moving average)
    - cosine schedule with warmup
    - checkpointing (full resume)
    - everything else your llm needs
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # distributed training setup
        self.is_distributed = args.distributed
        if self.is_distributed:
            # torchrun sets these env vars automatically
            dist.init_process_group(backend='nccl')
            self.device = torch.device(f"cuda:{args.local_rank}")
            torch.cuda.set_device(self.device)
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        
        # load data
        self._load_data()
        
        # create model
        self.cfg = self._get_config()
        self.model = MiniGPT(self.cfg).to(self.device)
        
        # compile first (if requested), THEN wrap with ddp
        if args.compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            if self.rank == 0:
                print("model compiled!!")
        
        # wrap for distributed training
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[args.local_rank])
        
        # optimizer (try fused adamw if available)
        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=0.1,
                fused=True
            )
            if self.rank == 0:
                print("🚀 using fused adamw (speed go brrr)")
        except:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=0.1
            )
            if self.rank == 0:
                print("fused adamw not available, using regular")
        
        # gradient accumulation
        self.accumulation_steps = args.accumulation_steps
        
        # lr scheduler (fixed to use effective steps)
        self.steps_per_epoch = len(self.train_loader)
        self.effective_steps_per_epoch = math.ceil(self.steps_per_epoch / self.accumulation_steps)
        total_steps = args.epochs * self.effective_steps_per_epoch
        
        # guard against negative T_max
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.warmup_steps
        )
        
        cosine_steps = max(1, total_steps - args.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            [warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_steps]
        )
        
        # mixed precision setup (only on cuda)
        self.use_amp = args.mixed_precision and self.device.type == "cuda"
        if self.use_amp:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.precision_dtype = torch.bfloat16
                if self.rank == 0:
                    print(" using bfloat16 (stable)")
            else:
                self.precision_dtype = torch.float16
                if self.rank == 0:
                    print(" using float16 (idk if your'e cooked!)")
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # ema (exponential moving average)
        self.ema_model = None
        if args.use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = args.ema_decay
        
        # tracking
        self.best_val_loss = float('inf')
        self.current_step = 0
    
    def _load_data(self):
        """load and tokenize with tiktoken"""
        if self.rank == 0:
            print(f"📖 loading data from {self.args.data}...")
        
        # read file
        with open(self.args.data, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # tokenize (bpe because character-level is so 2019)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = self.tokenizer.encode(text)
        
        # update vocab size
        self.args.vocab_size = self.tokenizer.n_vocab
        
        # convert to tensor
        data = torch.tensor(tokens, dtype=torch.long)
        
        # train/val split
        split_idx = int(0.9 * len(data))
        train_data, val_data = data[:split_idx], data[split_idx:]
        
        # create datasets
        train_dataset = TextDataset(train_data, self.args.context_length)
        val_dataset = TextDataset(val_data, self.args.context_length)
        
        # distributed sampler for train
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            train_sampler = self.train_sampler
            shuffle = False  # sampler handles shuffle
        else:
            self.train_sampler = None
            train_sampler = None
            shuffle = True
        
        # set num_workers based on cpu count (don't hardcode)
        num_workers = min(4, os.cpu_count() or 1)
        
        # create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # val loader (no sampler needed)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        if self.rank == 0:
            print(f" train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")
    
    def _get_config(self) -> Config:
        """create config from args"""
        preset_map = {
            'tiny': Config.tiny,
            'small': Config.small,
            'medium': Config.medium
        }
        
        if self.args.model_size in preset_map:
            cfg = preset_map[self.args.model_size]()
        else:
            cfg = Config()
        
        cfg.vocab_size = self.args.vocab_size
        cfg.context_length = self.args.context_length
        cfg.use_moe = self.args.use_moe
        
        return cfg
    
    def _create_ema_model(self):
        """create ema copy (slow weights)"""
        ema_model = MiniGPT(self.cfg).to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def _update_ema(self):
        """update ema weights (the slow blend)"""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            # handle ddp prefix if needed
            model_state = self.model.state_dict()
            ema_state = self.ema_model.state_dict()
            
            for name, param in model_state.items():
                # strip 'module.' prefix if present in ddp but not in ema
                ema_name = name.replace('module.', '') if name.startswith('module.') else name
                if ema_name in ema_state:
                    ema_state[ema_name].mul_(self.ema_decay).add_(param, alpha=1 - self.ema_decay)
    
    def train_epoch(self, epoch: int):
        """train for one epoch"""
        # set epoch for distributed sampler
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        self.optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # forward with mixed precision (only if on cuda)
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.precision_dtype):
                    if self.cfg.use_moe:
                        logits, aux_loss = self.model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.view(-1)
                        ) + aux_loss
                    else:
                        logits = self.model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.view(-1)
                        )
            else:
                # no amp on cpu
                if self.cfg.use_moe:
                    logits, aux_loss = self.model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    ) + aux_loss
                else:
                    logits = self.model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )
            
            # scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # update weights after accumulation steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # update ema and lr
                self._update_ema()
                self.scheduler.step()
                self.current_step += 1
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # log progress
            if batch_idx % 100 == 0 and self.rank == 0:
                print(f"  batch {batch_idx}/{len(self.train_loader)}, loss: {loss.item() * self.accumulation_steps:.4f}")
        
        # handle remaining gradients (if accumulation steps don't divide evenly)
        if len(self.train_loader) % self.accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self._update_ema()
            self.scheduler.step()
            self.current_step += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        """run validation"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.precision_dtype):
                    if self.cfg.use_moe:
                        logits, aux_loss = self.model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.view(-1)
                        ) + aux_loss
                    else:
                        logits = self.model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.view(-1)
                        )
            else:
                if self.cfg.use_moe:
                    logits, aux_loss = self.model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    ) + aux_loss
                else:
                    logits = self.model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def generate_sample(self, prompt: str, max_tokens: int = 100):
        """generate a sample for inspection"""
        # use ema for generation if available (more stable)
        if self.ema_model is not None:
            model = self.ema_model
        elif self.is_distributed:
            model = self.model.module
        else:
            model = self.model
        
        model.eval()
        
        # encode
        tokens = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([tokens], device=self.device)
        
        # generate
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        # decode
        return self.tokenizer.decode(generated[0].tolist())
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """save checkpoint (includes everything needed to resume)"""
        if self.rank != 0:
            return
        
        # get model state (handle ddp)
        if self.is_distributed:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'config': asdict(self.cfg),
            'tokenizer_name': "cl100k_base"
        }
        
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        # save latest
        torch.save(checkpoint, f"checkpoint_latest.pt")
        
        # save best
        if is_best:
            torch.save(checkpoint, f"checkpoint_best.pt")
        
        # save epoch checkpoint
        if epoch % self.args.save_every == 0:
            torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
    
    def load_checkpoint(self, path: str):
        """load checkpoint and resume training"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # handle ddp
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.ema_model is not None and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def train(self):
        """main training loop"""
        start_epoch = 0
        if self.args.resume:
            start_epoch, _ = self.load_checkpoint(self.args.resume)
            if self.rank == 0:
                print(f"resumed from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.args.epochs):
            # train
            train_loss = self.train_epoch(epoch)
            
            # validate
            val_loss = self.validate()
            
            # check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # log
            if self.rank == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"epoch {epoch+1}/{self.args.epochs} | "
                      f"train: {train_loss:.4f} | "
                      f"val: {val_loss:.4f} | "
                      f"lr: {lr:.2e} | "
                      f"{'best' if is_best else ''}")
                
                # generate sample every 5 epochs
                if (epoch + 1) % 5 == 0:
                    sample = self.generate_sample("The future of AI is")
                    print(f"\n sample:\n{sample}\n")
                
                # save checkpoint
                self.save_checkpoint(epoch + 1, val_loss, is_best)
        
        # cleanup distributed
        if self.is_distributed:
            dist.destroy_process_group()


def main():
    """parse args and train"""
    parser = argparse.ArgumentParser(description="minigpt - train a tiny transformer on your own data")
    
    # data
    parser.add_argument("--data", type=str, required=True, help="text file to train on")
    
    # model
    parser.add_argument("--model_size", type=str, default="small",
                        choices=['tiny', 'small', 'medium'],
                        help="model size: tiny (~10m), small (~30m), medium (~100m)")
    parser.add_argument("--context_length", type=int, default=256,
                        help="maximum context length")
    parser.add_argument("--use_moe", action="store_true",
                        help="enable mixture of experts (expensive)")
    
    # training
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size per gpu")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="learning rate warmup steps")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="gradient accumulation steps")
    
    # optimization
    parser.add_argument("--mixed_precision", action="store_true",
                        help="use mixed precision (cuda only)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model (torch 2.0+)")
    parser.add_argument("--use_ema", action="store_true",
                        help="use exponential moving average")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="ema decay rate")
    
    # distributed
    parser.add_argument("--distributed", action="store_true",
                        help="use distributed training (via torchrun)")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local rank (set by torchrun)")
    
    # checkpointing
    parser.add_argument("--resume", type=str, default=None,
                        help="resume from checkpoint")
    parser.add_argument("--save_every", type=int, default=5,
                        help="save checkpoint every n epochs")
    
    args = parser.parse_args()
    
    # train
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
#github: https://github.com/Aranya-Marjara
