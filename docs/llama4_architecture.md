# Llama 4: The Latest Public Llama Model

> As of Nov 2026, **Llama 4** (released 5 April 2025) is the newest model Meta has published under the *Llama* name.
> Meta's 2026 open-weight releases (Muse Glimmer, Muse Spark Open) use a different brand. Reports of a
> "Llama 4.5" are unverified. This repo uses Llama 4 as the reference architecture.

## 1. The family

| Model | Active params | Experts | Total params | Context | Status |
|---|---|---|---|---|---|
| **Llama 4 Scout** | 17B | 16 routed + 1 shared | ~109B | 10M tokens | open weights |
| **Llama 4 Maverick** | 17B | 128 routed + 1 shared | ~400B | 1M tokens | open weights |
| **Llama 4 Behemoth** | 288B | 16 | ~2T | — | teacher model, never released |

"Active params" is the amount of the network each token actually passes through. Because it is a Mixture of Experts, most weights sit idle for any given token.

Shared shape of Scout and Maverick (from the Hugging Face config): hidden size 5120, 48 layers, 40 query heads,
8 key/value heads, head dim 128, vocabulary of 202,048 tokens, RoPE θ = 500,000.

## 2. The architecture, piece by piece

Llama 4 is a **decoder-only transformer**. Text (and image patches) become a sequence of vectors, which pass through a stack of blocks. Each block **adds** its result to a running *residual stream*:

```
tokens ──► embedding ──► [ Block × 48 ] ──► RMSNorm ──► output projection ──► logits over 202k tokens

Block:   h = h + Attention( RMSNorm(h) )
         h = h + FFN_or_MoE( RMSNorm(h) )
```

### 2.1 RMSNorm (pre-norm)
`RMSNorm(x) = x / sqrt(mean(x²) + ε) · g`. It rescales each vector to unit root-mean-square and has a learned per-channel gain `g`. Unlike LayerNorm, it does not subtract the mean. The norm sits *before* each sub-layer, which keeps the residual stream clean and makes deep stacks stable to train.

### 2.2 Grouped-Query Attention (GQA)
Each token produces a query, key and value (`q = xWq, k = xWk, v = xWv`). There are 40 query heads but only 8 K/V heads, so 5 query heads share each K/V head. This makes the KV-cache 5× smaller at inference with almost no loss in quality.
`Attention = softmax(q kᵀ / √d + mask) · v`, followed by an output projection `Wo`.

### 2.3 iRoPE: interleaved RoPE / NoPE layers (new in Llama 4)
- **RoPE layers** (3 out of every 4): queries and keys are *rotated* by an angle proportional to their position, so `q·k` depends on relative distance. These layers use **chunked attention**: a token only sees tokens inside its own 8,192-token chunk. That makes them cheap and local.
- **NoPE layers** (every 4th layer): **no positional encoding at all**, and full causal attention over the whole context. The causal mask alone gives the model enough information to infer order. These layers handle long-range retrieval.
- **Attention temperature tuning**: on NoPE layers, queries are scaled by `1 + s·log(⌊(pos+1)/8192⌋ + 1)`. This keeps attention sharp at very long positions and is what lets Scout reach a 10M context.
- **QK-norm**: on RoPE layers, `q` and `k` are RMS-normalized (with no learned weight) before the dot product, which keeps attention logits bounded.

"i" stands for *interleaved*, and the design aims at infinite context length.

### 2.4 SwiGLU feed-forward
`FFN(x) = W2( SiLU(W1 x) ⊙ W3 x )`. A gated MLP: `W1` produces a gate, `W3` produces values, and `W2` projects back down to the model width.

### 2.5 Mixture of Experts (new in Llama 4)
- In an MoE layer there is a **shared expert** that every token uses, plus **N routed experts**. A small **router** (`x·Wr`) scores the experts for each token. The token is sent to the **top-1** expert, and the router's **sigmoid** score scales the expert's input.
- `out = SharedFFN(x) + Expert_top1( x · σ(router_score) )`
- **Maverick alternates** dense FFN layers and MoE layers. In **Scout**, every layer is MoE.
- Result: 400B parameters of knowledge, but only 17B worth of compute per token.

### 2.6 Early-fusion multimodality
Image patches pass through a vision encoder (based on MetaCLIP) and are projected into the same token space. They are then fed **into the same sequence** as the text from the very first layer, and the whole thing is pre-trained jointly. (Our mini model is text-only.)

## 3. How Llama 4 was trained

1. **Pre-training**: next-token prediction (cross-entropy loss) on roughly 30T+ tokens of text, image and video data in 200 languages (the model card lists ~40T for Scout and ~22T for Maverick). Training ran in **FP8** precision. **MetaP** was used to pick per-layer learning rates and init scales that transfer across model sizes. Maverick was also *co-distilled* from Behemoth.
2. **Mid-training**: long-context extension using specialized datasets.
3. **Post-training**: lightweight **SFT** (on hard examples only), then **online RL**, then lightweight **DPO**.

### The core loop (same for Llama 4 and our mini model)
```
x, y      = batch[:, :-1], batch[:, 1:]          # inputs and next-token targets
logits    = model(x)                              # forward pass
loss      = cross_entropy(logits, y)              # how surprised was the model?
optimizer.zero_grad()
loss.backward()                                   # backprop: chain rule computes dLoss/dW for every weight
clip_grad_norm_(params, 1.0)                      # guard against exploding gradients
optimizer.step()                                  # AdamW: W ← W − lr · m̂/(√v̂+ε) − lr·λ·W
lr = warmup_then_cosine(step)
```

## 4. Real Llama 4 vs. Mini Llama 4 (this repo)

| Component | Llama 4 Maverick | Mini Llama 4 (`mini_llama4/`) |
|---|---|---|
| Tokenizer | 202k BPE (tiktoken-based) | 65 characters |
| Width / layers | 5120 / 48 | 128 / 4 |
| Heads (Q / KV) | 40 / 8 | 4 / 2 (GQA kept) |
| Norm | RMSNorm, pre-norm | same (`RMSNorm`) |
| Positional | iRoPE: RoPE+chunked (8192) / NoPE every 4th | same pattern, chunk = 32 (`Attention`) |
| QK-norm | on RoPE layers | same (`qk_norm`) |
| Temperature tuning | NoPE layers, floor 8192 | same, floor 32 |
| FFN | SwiGLU | same (`FeedForward`) |
| MoE | 128 routed + shared, top-1, sigmoid, every 2nd layer | 4 routed + shared, top-1, sigmoid, every 2nd layer (`MoE`) |
| Vision | early fusion | — (text only) |
| Precision | FP8 / BF16 | FP32 |
| Params | ~400B total / 17B active | ~1.4M total |
| Data | ~22T tokens | TinyShakespeare (~1M chars) |

Layer pattern in the mini model: `0: dense/RoPE · 1: MoE/RoPE · 2: dense/RoPE · 3: MoE/NoPE`.

## Sources
- Meta AI blog, "The Llama 4 herd" (April 2025), ai.meta.com/blog/llama-4-multimodal-intelligence
- Llama 4 model card and `transformers` `Llama4TextConfig` / `modeling_llama4.py`
- [Llama (language model), Wikipedia](https://en.wikipedia.org/wiki/Llama_(language_model))
