"""Mini Llama 4: a small, readable, text-only re-implementation of Meta's Llama 4.

Kept from the real architecture:
  * pre-norm decoder-only transformer with RMSNorm and a residual stream
  * Grouped-Query Attention (fewer K/V heads than Q heads)
  * iRoPE: RoPE layers with chunked (local) attention + QK-norm,
           interleaved with NoPE layers that use global attention and temperature tuning
  * SwiGLU feed-forward
  * Mixture-of-Experts layers (shared expert + top-1 routed expert, sigmoid gate),
    interleaved with dense layers
Dropped: vision encoder (early fusion), FP8, tensor/expert parallelism.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2)) * g  -- normalizes the size of each vector, no mean-centering."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def rope_tables(head_dim: int, seq_len: int, theta: float):
    """Precompute cos/sin of the rotation angle for every (position, frequency) pair."""
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)  # (hd/2,)
    angles = torch.outer(torch.arange(seq_len).float(), freqs)                  # (T, hd/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each consecutive pair (x0,x1),(x2,x3),... by a position-dependent angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


def qk_norm(x, eps: float):
    """Llama 4's 'L2Norm' on queries/keys: RMS-normalize each head vector, no learned weight.
    Stops attention logits from blowing up as weights grow."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class Attention(nn.Module):
    def __init__(self, cfg: Config, use_rope: bool):
        super().__init__()
        self.cfg = cfg
        self.use_rope = use_rope
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.hd, cfg.dim, bias=False)

    def mask(self, T: int, device):
        """True = allowed. Causal everywhere; RoPE layers are also restricted to their chunk."""
        i = torch.arange(T, device=device)
        allowed = i[None, :] <= i[:, None]                       # causal: no peeking ahead
        if self.use_rope:
            chunk = i // self.cfg.attn_chunk
            allowed &= chunk[None, :] == chunk[:, None]          # local chunked attention
        return allowed

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)  # (B, H, T, hd)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)     # (B, KV, T, hd)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)

        if self.use_rope:
            q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
            q, k = qk_norm(q, self.cfg.norm_eps), qk_norm(k, self.cfg.norm_eps)  # QK-norm
        else:
            # NoPE layer: temperature tuning -- sharpen attention further into the sequence
            pos = torch.arange(T, device=x.device).float()
            temp = torch.log(torch.floor((pos + 1) / self.cfg.attn_temp_floor) + 1)
            q = q * (1 + self.cfg.attn_temp_scale * temp)[None, None, :, None]

        # GQA: each K/V head is shared by n_heads / n_kv query heads
        rep = self.n_heads // self.n_kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hd)             # (B, H, T, T)
        scores = scores.masked_fill(~self.mask(T, x.device), float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)                # concat heads
        return self.wo(out)


class FeedForward(nn.Module):
    """SwiGLU: w2( silu(w1 x) * w3 x )."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, dim, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Llama 4 style MoE: every token goes through the shared expert, plus its top-k routed
    expert(s). The router score is a sigmoid and scales the expert's *input*."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(FeedForward(cfg.dim, cfg.ffn_hidden) for _ in range(cfg.n_experts))
        self.shared_expert = FeedForward(cfg.dim, cfg.ffn_hidden)
        self.last_routing = None  # (N, top_k) expert ids from the last forward, for inspection

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)                                    # (N, D) tokens
        logits = self.router(flat)                                 # (N, E)
        top_val, top_idx = logits.topk(self.top_k, dim=-1)         # (N, k)
        gate = torch.sigmoid(top_val)                              # (N, k)
        self.last_routing = top_idx.detach()

        out = self.shared_expert(flat)
        for e, expert in enumerate(self.experts):
            token_ids, slot = (top_idx == e).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            routed_in = flat[token_ids] * gate[token_ids, slot, None]
            out = out.index_add(0, token_ids, expert(routed_in))
        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_id: int):
        super().__init__()
        self.use_rope = not cfg.is_nope_layer(layer_id)
        self.is_moe = cfg.is_moe_layer(layer_id)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg, use_rope=self.use_rope)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = MoE(cfg) if self.is_moe else FeedForward(cfg.dim, cfg.ffn_hidden)

    def forward(self, h, cos, sin):
        h = h + self.attn(self.attn_norm(h), cos, sin)   # residual 1
        h = h + self.ffn(self.ffn_norm(h))               # residual 2
        return h


class MiniLlama4(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns logits (B, T, V) and, if targets given, the loss."""
        h = self.tok_embeddings(idx)
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)
        logits = self.output(self.norm(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = 20):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.seq_len:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                kth = logits.topk(top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = torch.multinomial(logits.softmax(-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
