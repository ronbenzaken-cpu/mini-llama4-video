from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters of Mini Llama 4.

    Real Llama 4 Maverick uses dim=5120, 48 layers, 40 heads, 8 KV heads,
    128 routed experts and a 202k-token vocabulary. We keep the same *shape*
    of the architecture but shrink every number so it trains on a laptop CPU.
    """

    vocab_size: int = 65        # char-level tokenizer (TinyShakespeare has 65 chars)
    dim: int = 128              # width of the residual stream
    n_layers: int = 4
    n_heads: int = 4            # query heads
    n_kv_heads: int = 2         # key/value heads (Grouped-Query Attention)
    ffn_hidden: int = 256       # hidden size of each SwiGLU feed-forward / expert
    n_experts: int = 4          # routed experts in an MoE layer
    top_k: int = 1              # Llama 4 routes each token to 1 routed expert
    moe_every: int = 2          # every 2nd layer is MoE, others dense (like Maverick)
    nope_every: int = 4         # every 4th layer has NO positional encoding (iRoPE)
    attn_chunk: int = 32        # RoPE layers attend only inside local chunks
    seq_len: int = 128          # context length used in training
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    # Attention temperature tuning on NoPE layers (helps long-context generalization).
    # Real model: floor_scale=8192. Smaller here so the effect is visible at seq_len=128.
    attn_temp_floor: float = 32.0
    attn_temp_scale: float = 0.1

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def is_moe_layer(self, i: int) -> bool:
        return (i + 1) % self.moe_every == 0

    def is_nope_layer(self, i: int) -> bool:
        return (i + 1) % self.nope_every == 0
