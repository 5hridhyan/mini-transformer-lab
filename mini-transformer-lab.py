#!/usr/bin/env python3
"""
mini-transformer-lab.py

A small transformer-based language model I built to actually *understand*
how these big LLMs work under the hood. It's not about scale — it's about clarity.
Runs fully in PyTorch, no fancy trainer, just good old loops.

This version:
- Includes attention masking, learned or sinusoidal positions
- Handles top-k and top-p sampling correctly
- Saves and loads checkpoints
- Trains on a text file you provide
- Can generate from command line
"""
# NOTE: This used to be implemented differently but caused unstable loss;
# simplified after debugging.


import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
import argparse
import os


#  Basic Config 
@dataclass
class ModelConfig:
    vocab_size: int = 5000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    context_length: int = 256
    use_learned_positions: bool = True


# Positional Encodings 
class SinusoidalPositionalEmbedding(nn.Module):
    """Classic transformer sine-cosine positions (from 'Attention is All You Need')."""
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        # Create the positional encoding matrix once
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # Return positional encodings for the sequence length of x
        return self.pe[:, :x.size(1)]


# Core Transformer Blocks 
class MultiHeadAttention(nn.Module):
    """Causal self-attention (no lookahead)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self._mask_cache = {}

    def _causal_mask(self, seq_len, device):
        key = (seq_len, device)
        if key not in self._mask_cache:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x):
        B, T, D = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._causal_mask(T, x.device)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# Main Mini Transformer
class MiniTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.use_learned_positions:
            self.pos_embed = nn.Embedding(cfg.context_length, cfg.d_model)
        else:
            self.pos_embed = SinusoidalPositionalEmbedding(cfg.d_model, cfg.context_length)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        B, T = tokens.size()
        if T > self.cfg.context_length:
            # If sequence is too long, just use the last context_length tokens
            tokens = tokens[:, -self.cfg.context_length:]
            T = self.cfg.context_length
            
        tok = self.token_embed(tokens)
        if self.cfg.use_learned_positions:
            pos = torch.arange(T, device=tokens.device).unsqueeze(0)
            pos = self.pos_embed(pos)
        else:
            # For sinusoidal, we need to create a dummy tensor to get the right shape
            dummy = torch.zeros(1, T, self.cfg.d_model, device=tokens.device)
            pos = self.pos_embed(dummy)
        x = self.drop(tok + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None):
        """Generate new tokens autoregressively from a given prompt."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = prompt[:, -self.cfg.context_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # --- top-k sampling ---
            if top_k is not None:
                vocab_size = logits.size(-1)
                k = min(top_k, vocab_size)
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # --- top-p (nucleus) sampling ---
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum = probs.cumsum(dim=-1)
                mask = cumsum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = -float('Inf')
                logits.scatter_(1, sorted_idx, sorted_logits)

            # --- sample next token ---
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat((prompt, next_token), dim=1)

        return prompt


# Simple Character Tokenizer
class CharTokenizer:
    def __init__(self, text=None, vocab=None):
        if text is not None:
            vocab = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(vocab)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(vocab)
        elif vocab is not None:
            self.stoi = {ch: i for i, ch in enumerate(vocab)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(vocab)
        else:
            raise ValueError("Need either text or vocab to create tokenizer")

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]  # Use get for safety

    def decode(self, t):
        return ''.join([self.itos.get(i, '?') for i in t])

    def get_vocab(self):
        return list(self.stoi.keys())


# Training Helpers
def get_batch(data, bs, ctx_len, device):
    """Get a random batch of training data."""
    # Make sure we don't go out of bounds
    max_start = len(data) - ctx_len - 1
    if max_start <= 0:
        raise ValueError("Data too short for the given context length")
        
    ix = torch.randint(0, max_start, (bs,))
    x = torch.stack([data[i:i+ctx_len] for i in ix])
    y = torch.stack([data[i+1:i+1+ctx_len] for i in ix])
    return x.to(device), y.to(device)


def estimate_loss(model, data, bs, ctx_len, device, n_batches=10):
    """Estimate the loss on validation data."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(n_batches):
            xb, yb = get_batch(data, bs, ctx_len, device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / n_batches


def save_checkpoint(model, opt, tokenizer, epoch, path="mini_llm_checkpoint.pt"):
    """Save everything needed to resume training or generate later"""
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "tokenizer_vocab": tokenizer.get_vocab(),
        "model_config": model.cfg.__dict__,
        "epoch": epoch
    }, path)
    print(f"[Checkpoint saved @ {path}]")


def load_checkpoint(path, device):
    """Load model, tokenizer, and config from checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    # Recreate config
    cfg_dict = checkpoint["model_config"]
    cfg = ModelConfig(
        vocab_size=cfg_dict["vocab_size"],
        d_model=cfg_dict["d_model"],
        n_heads=cfg_dict["n_heads"],
        n_layers=cfg_dict["n_layers"],
        d_ff=cfg_dict["d_ff"],
        dropout=cfg_dict["dropout"],
        context_length=cfg_dict["context_length"],
        use_learned_positions=cfg_dict["use_learned_positions"]
    )
    
    # Recreate tokenizer
    tokenizer = CharTokenizer(vocab=checkpoint["tokenizer_vocab"])
    
    # Recreate model
    model = MiniTransformer(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    
    return model, tokenizer, cfg, checkpoint


# Training Loop
def train_model(args):
    """The main training function - fixed to actually train properly!"""
    print(f"Loading data from {args.data}...")
    text = open(args.data, encoding="utf-8").read()
    print(f"Loaded {len(text)} characters")
    
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        context_length=args.context_length
    )
    model = MiniTransformer(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training Mini LLM on '{args.data}' with {tok.vocab_size} tokens")
    print(f"Context length: {cfg.context_length}, Model dim: {cfg.d_model}")
    print("=" * 60)

    # If resuming from checkpoint
    start_epoch = 1
    if args.resume:
        print(f"Resuming from {args.resume}...")
        model, tok, cfg, checkpoint = load_checkpoint(args.resume, args.device)
        opt.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Calculate how many batches we can get from our training data
    steps_per_epoch = max(1, len(train_data) // (args.batch_size * cfg.context_length))
    print(f"Training with {steps_per_epoch} steps per epoch")
    print(f"Total training steps: {steps_per_epoch * args.epochs}")

    # Training loop that actually trains on multiple batches per epoch!
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        total_train_loss = 0
        
        # Train for one epoch (multiple batches)
        model.train()
        for step in range(steps_per_epoch):
            xb, yb = get_batch(train_data, args.batch_size, cfg.context_length, args.device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            opt.step()
            total_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / steps_per_epoch
        
        # Validation
        val_loss = estimate_loss(model, val_data, args.batch_size, cfg.context_length, args.device)
        epoch_time = time.time() - epoch_start
        
        print(f"epoch {epoch:02d}/{args.epochs} | train {avg_train_loss:.4f} | val {val_loss:.4f} | time {epoch_time:.1f}s")

        # Save checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:
            save_path = f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(model, opt, tok, epoch, save_path)

    # Final sample generation
    print("\n" + "="*60)
    print("Final generation sample:")
    print("="*60)
    prompt = "The meaning of life is"
    prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=args.device)
    out = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8, top_k=50)
    print(f"Prompt: {prompt}")
    print(f"Generated: {tok.decode(out[0].tolist())}")


# Generation Function 
def generate_text(args):
    """Load a trained model and generate some text"""
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer, cfg, _ = load_checkpoint(args.checkpoint, args.device)
    
    print(f"Model loaded: {cfg.vocab_size} vocab, {cfg.context_length} context")
    
    # Encode prompt
    prompt = args.prompt
    if len(prompt) > cfg.context_length:
        prompt = prompt[-cfg.context_length:]
        print(f"Note: Prompt truncated to last {cfg.context_length} chars")
    
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)
    
    print("\n" + "="*50)
    print(f"PROMPT: {prompt}")
    print("GENERATING..." + "="*50)
    
    # Generate
    start_time = time.time()
    output_ids = model.generate(
        prompt_ids, 
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    gen_time = time.time() - start_time
    
    # Decode and print
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)
    print("="*50)
    print(f"Generated {args.max_new_tokens} tokens in {gen_time:.2f}s")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")


# Main 
def main():
    parser = argparse.ArgumentParser(description="Mini LLM - Train or generate text")
    subparsers = parser.add_subparsers(dest='command', help='What to do')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument("--data", type=str, required=True, help="Text file to train on")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--context_length", type=int, default=256, help="Context length")
    train_parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate command  
    gen_parser = subparsers.add_parser('generate', help='Generate text from trained model')
    gen_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    gen_parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Starting text")
    gen_parser.add_argument("--max_new_tokens", type=int, default=100, help="Tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    gen_parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    gen_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'generate':
        generate_text(args)
    else:
        print("Please specify 'train' or 'generate'")
        print("\nExamples:")
        print("  python mini_llm.py train --data my_book.txt --epochs 20")
        print("  python mini_llm.py generate --checkpoint checkpoint_epoch_10.pt --prompt 'Once upon a'")
        print("  python mini_llm.py train --data my_book.txt --resume checkpoint_epoch_5.pt --epochs 10")


if __name__ == "__main__":
    main()
