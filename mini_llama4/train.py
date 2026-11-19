"""Train Mini Llama 4 on TinyShakespeare.

    python -m mini_llama4.train --steps 300
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from .config import Config
from .data import CharTokenizer, Dataset, load_text
from .model import MiniLlama4

OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay down to min_ratio * max_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_val_loss(model, ds, cfg, batch_size, device, iters=10):
    model.eval()
    losses = [model(*ds.get_batch("val", batch_size, cfg.seq_len, device))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data ----
    text = load_text()
    tok = CharTokenizer(text)
    ds = Dataset(text, tok)
    cfg = Config(vocab_size=tok.vocab_size)

    # ---- model + optimizer ----
    model = MiniLlama4(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params/1e6:.2f}M  layers: "
          + ", ".join(f"{i}:{'MoE' if b.is_moe else 'dense'}/{'RoPE' if b.use_rope else 'NoPE'}"
                      for i, b in enumerate(model.layers)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    OUT_DIR.mkdir(exist_ok=True)
    log = open(OUT_DIR / "loss_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "lr", "train_loss", "val_loss", "grad_norm"])

    # ---- training loop ----
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = ds.get_batch("train", args.batch_size, cfg.seq_len, device)  # inputs / next-token targets
        logits, loss = model(x, y)                                          # forward pass
        opt.zero_grad(set_to_none=True)                                     # clear old gradients
        loss.backward()                                                     # backprop: dLoss/dWeight
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip exploding grads
        opt.step()                                                          # AdamW update

        val = ""
        if step % args.eval_every == 0 or step == args.steps - 1:
            val = estimate_val_loss(model, ds, cfg, args.batch_size, device)
            print(f"step {step:5d} | lr {lr:.2e} | train {loss.item():.3f} | val {val:.3f} "
                  f"| grad {grad_norm:.2f} | {time.time()-t0:.0f}s")
        writer.writerow([step, f"{lr:.6g}", f"{loss.item():.4f}", val if val == "" else f"{val:.4f}",
                         f"{float(grad_norm):.4f}"])
    log.close()

    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "chars": tok.chars},
               OUT_DIR / "mini_llama4.pt")

    # ---- sample ----
    model.eval()
    start = torch.tensor([tok.encode("ROMEO:\n")], device=device)
    print("\n--- sample ---")
    print(tok.decode(model.generate(start, max_new_tokens=300)[0].tolist()))


if __name__ == "__main__":
    main()
