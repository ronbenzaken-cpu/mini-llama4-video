"""Real numbers from our Mini Llama 4 (written by video/make_trace.py). Scenes read ONLY from here.

    from video.common.data import TRACE
    TRACE["tokens"]["chars"]                 # ['T','o',' ','b','e',' ','o','r']
    TRACE["tokens"]["x_ids"], ["y_ids"]      # inputs and next-char targets
    TRACE["config"]                          # dim, n_layers, n_heads, n_kv_heads, n_experts, vocab_size, n_params...
    TRACE["layers"]                          # [{'index','moe','rope'}] x4
    TRACE["vocab"], TRACE["excerpt"]         # 65 chars, Shakespeare excerpt
    TRACE["embedding_first8"]                # 8 tokens x first 8 embedding dims (trained)
    TRACE["attention_layer0_head0"]          # 8x8 causal attention (trained, layer 0 = RoPE + chunked)
    TRACE["moe_routing_first_moe_layer"]     # expert id per token (layer 1)
    TRACE["untrained"]                       # last_token_probs[65], target_id, p_target, loss, grad_norms{}
    TRACE["trained"]                         # last_token_probs[65], p_target
    TRACE["loss_curve"]                      # [[step, train_loss], ...] 300 steps
    TRACE["samples"]                         # {'untrained': str, 'trained': str}
"""

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "assets" / "trace.json"
TRACE = json.loads(_PATH.read_text(encoding="utf-8"))


def show_char(c: str) -> str:
    """Make whitespace visible on screen."""
    return {" ": "·",  # Consolas has no ␣ glyph; middle dot renders
            "\n": "↵"}.get(c, c)
