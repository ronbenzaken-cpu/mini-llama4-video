# Mini Llama 4: training a transformer from scratch

The goal of this project is to explain, line by line, how a modern LLM is trained. It covers the forward pass, the loss, backprop and the optimizer step. The explanation will be a 3Blue1Brown-style animated video, and the video plan comes later.

- `docs/llama4_architecture.md` describes Meta's Llama 4 (the latest public Llama) and maps it to the mini version.
- `mini_llama4/` is a small (~1.4M param) text-only Llama 4: RMSNorm, GQA, iRoPE (RoPE+chunked / NoPE), QK-norm, SwiGLU, MoE.
- `tests/` holds the unit tests.

## Quick start
```bash
python -m venv .venv
.venv/Scripts/activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python -m pytest tests/
python -m mini_llama4.train --steps 1000   # downloads TinyShakespeare, trains on CPU
```
Training writes `checkpoints/loss_log.csv` and `checkpoints/mini_llama4.pt`, and prints a generated sample.
