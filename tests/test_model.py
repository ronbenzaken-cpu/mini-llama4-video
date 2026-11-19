import torch

from mini_llama4 import Config, MiniLlama4
from mini_llama4.model import Attention, apply_rope, rope_tables


def small_cfg():
    return Config(vocab_size=50, dim=32, n_heads=4, n_kv_heads=2, ffn_hidden=64,
                  n_layers=4, seq_len=16, attn_chunk=8)


def test_layer_pattern():
    model = MiniLlama4(small_cfg())
    assert [b.is_moe for b in model.layers] == [False, True, False, True]
    assert [b.use_rope for b in model.layers] == [True, True, True, False]


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = MiniLlama4(cfg)
    x = torch.randint(cfg.vocab_size, (3, cfg.seq_len))
    logits, loss = model(x, x)
    assert logits.shape == (3, cfg.seq_len, cfg.vocab_size)
    assert torch.isfinite(loss)
    # untrained model ~ uniform guess: loss ≈ ln(V)
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.5


def test_moe_routing_shape():
    cfg = small_cfg()
    model = MiniLlama4(cfg)
    model(torch.randint(cfg.vocab_size, (2, cfg.seq_len)))
    routing = model.layers[1].ffn.last_routing
    assert routing.shape == (2 * cfg.seq_len, cfg.top_k)
    assert routing.max() < cfg.n_experts


def test_masks():
    cfg = small_cfg()
    rope_mask = Attention(cfg, use_rope=True).mask(16, "cpu")
    nope_mask = Attention(cfg, use_rope=False).mask(16, "cpu")
    assert not rope_mask[3, 4] and not nope_mask[3, 4]   # causal
    assert not rope_mask[9, 2] and nope_mask[9, 2]        # chunked only in RoPE layers


def test_rope_preserves_norm():
    cos, sin = rope_tables(8, 16, 10000.0)
    x = torch.randn(1, 1, 16, 8)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_loss_decreases_when_overfitting():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = MiniLlama4(cfg)
    x = torch.randint(cfg.vocab_size, (4, cfg.seq_len + 1))
    inp, tgt = x[:, :-1], x[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(50):
        _, loss = model(inp, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.5
