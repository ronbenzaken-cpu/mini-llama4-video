"""Run Mini Llama 4 once and save every real number the video shows -> video/assets/trace.json.

    .venv/Scripts/python video/make_trace.py

Needs checkpoints/mini_llama4.pt and checkpoints/loss_log.csv (python -m mini_llama4.train --steps 300).
"""

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_llama4 import Config, MiniLlama4  # noqa: E402
from mini_llama4.data import CharTokenizer, load_text  # noqa: E402

TEXT_IN = "To be or"        # the 8 tokens shown in Act 2/3 (x)
TEXT_FULL = "To be or "     # one more char so every x token has a target (y)
OUT = ROOT / "video" / "assets" / "trace.json"


def r(t, nd=4):
    return [round(float(v), nd) for v in t.flatten().tolist()] if t.dim() <= 1 else [r(row, nd) for row in t]


def forward_capture(model, idx):
    """Forward pass that also returns attention of layer 0 and routing of the first MoE layer."""
    cap = {}
    attn0 = model.layers[0].attn
    orig_softmax = torch.Tensor.softmax

    def grab_softmax(self, *a, **k):  # capture the first softmax (layer 0 attention)
        out = orig_softmax(self, *a, **k)
        if "attn" not in cap and self.dim() == 4:
            cap["attn"] = out.detach()
        return out

    torch.Tensor.softmax = grab_softmax
    try:
        logits, _ = model(idx)
    finally:
        torch.Tensor.softmax = orig_softmax
    moe = next(b.ffn for b in model.layers if b.is_moe)
    cap["routing"] = moe.last_routing.view(-1).tolist()
    del attn0
    return logits, cap


@torch.no_grad()
def sample(model, tok, prompt, n, seed):
    torch.manual_seed(seed)
    out = model.generate(torch.tensor([tok.encode(prompt)]), max_new_tokens=n)
    return tok.decode(out[0].tolist())


def main():
    text = load_text()
    tok = CharTokenizer(text)
    ckpt = torch.load(ROOT / "checkpoints" / "mini_llama4.pt", map_location="cpu")
    cfg = Config(**ckpt["config"])

    torch.manual_seed(0)
    untrained = MiniLlama4(cfg).eval()
    trained = MiniLlama4(cfg)
    trained.load_state_dict(ckpt["model"])
    trained.eval()

    ids_full = tok.encode(TEXT_FULL)
    x = torch.tensor([ids_full[:-1]])
    y = torch.tensor([ids_full[1:]])

    # ---- untrained model: what the video shows "before training" ----
    logits_u, _ = forward_capture(untrained, x)
    probs_u = logits_u[0, -1].softmax(-1)
    loss_u = F.cross_entropy(logits_u[0], y[0])
    p_target_u = probs_u[y[0, -1]].item()

    # gradient norms for one backward pass on the untrained model (scene 3.8)
    untrained.train()
    _, loss_b = untrained(x, y)
    loss_b.backward()
    groups = {"embedding": untrained.tok_embeddings, "output": untrained.output}
    for i, layer in enumerate(untrained.layers):
        groups[f"layer{i}"] = layer
    grad_norms = {k: round(float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None))), 4)
                  for k, m in groups.items()}

    # ---- trained model: attention + routing look meaningful after training ----
    logits_t, cap = forward_capture(trained, x)
    probs_t = logits_t[0, -1].softmax(-1)

    with open(ROOT / "checkpoints" / "loss_log.csv") as f:
        rows = list(csv.DictReader(f))
    loss_curve = [[int(rw["step"]), float(rw["train_loss"])] for rw in rows]

    n_params = sum(p.numel() for p in trained.parameters())
    trace = {
        "config": {**ckpt["config"], "head_dim": cfg.head_dim, "n_params": n_params},
        "layers": [{"index": i, "moe": b.is_moe, "rope": b.use_rope} for i, b in enumerate(trained.layers)],
        "vocab": tok.chars,
        "excerpt": text[:400],
        "tokens": {"text": TEXT_IN, "chars": list(TEXT_IN), "x_ids": x[0].tolist(), "y_ids": y[0].tolist(),
                   "y_chars": [tok.itos[i] for i in y[0].tolist()]},
        "embedding_first8": r(trained.tok_embeddings.weight[x[0], :8], 3),
        "attention_layer0_head0": r(cap["attn"][0, 0], 3),
        "moe_routing_first_moe_layer": cap["routing"],
        "untrained": {"last_token_probs": r(probs_u, 5), "target_id": y[0, -1].item(),
                      "p_target": round(p_target_u, 5), "loss": round(loss_u.item(), 4),
                      "grad_norms": grad_norms},
        "trained": {"last_token_probs": r(probs_t, 5), "p_target": round(probs_t[y[0, -1]].item(), 5)},
        "loss_curve": loss_curve,
        "samples": {"untrained": sample(untrained.eval(), tok, "ROMEO:\n", 160, 1),
                    "trained": sample(trained, tok, "ROMEO:\n", 160, 1)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(trace, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}  (loss untrained {loss_u.item():.3f}, p_target {p_target_u:.4f}, params {n_params})")


if __name__ == "__main__":
    main()
