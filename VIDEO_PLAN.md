# Video Plan: "How a Llama Learns"

Three acts: **(1) the real Llama 4 network → (2) our Mini Llama 4 → (3) the training code, line by line.**

## 0. Quiz answer (Week 6, last question)

| Field | Answer |
|---|---|
| **Topic** | How a large language model (Meta's Llama 4) is built and trained: architecture, then the forward pass, loss, backpropagation and weight update in real code |
| **After one minute the viewer understands that…** | *An LLM like Llama is a stack of attention and expert layers that predicts the next token, and training is one loop repeated thousands of times: predict, measure the error with the loss, backprop to find each weight's blame, nudge the weights.* |
| **Stack** | Python 3.11 · Manim Community Edition (3Blue1Brown-style) · manim-voiceover + gTTS (narration) · PyTorch (real numbers from our mini model) · Claude Code · git/GitHub |

## 1. Lab constraints that shape the plan
- The instructor plays **the start** of each video, and the takeaway has to land **within the first minute**. So the video opens with a **10 s cold open** that states the whole idea; the three acts then unpack it. Total target: **~3 min**.
- Narration throughout. All visuals are drawn in code. No AI-generated images.
- **One scene = one file.** Small steps, preview with `-ql`, commit after each step (*what + why*).
- Renders go to `renders/render-NN.mp4`. Never overwrite one. A partial video is fine.
- `SCRIPT.md` at the repo root holds every scene's visual and narration.

## 2. The flow

| Act | Goal: the viewer understands… | Scenes | Time |
|---|---|---|---|
| **0 Cold open** | the one-sentence takeaway | 1 | 10 s |
| **1 Llama 4** | what the real network looks like and why it is built that way | 6 | ~60 s |
| **2 Mini Llama 4** | we kept the same design and shrank the numbers, so it trains on a laptop | 3 | ~35 s |
| **3 Code, line by line** | what each line of a training step does inside the network | 9 | ~85 s |
| **Outro** | before/after text, the takeaway again | 1 | 10 s |

---

## Act 0 — Cold open (1 scene)
| # | File | Visual | Narration |
|---|---|---|---|
| 0.1 | `act0/s01_cold_open.py` | "To be or not to ___", a glowing box guesses the next word wrong, then the loop arrow *predict → loss → backprop → update* spins and the guess becomes right | "A language model learns by guessing the next word, measuring how wrong it was, and nudging its weights. Thousands of times. Let's see how." |

**Needs:** the loop diagram (reused in Act 3 and the outro), a token-row component.

## Act 1 — The real Llama 4 (6 scenes)
Source of facts: `docs/llama4_architecture.md`.

| # | File | Visual | Narration (gist) |
|---|---|---|---|
| 1.1 | `act1/s01_family.py` | Scout / Maverick / Behemoth as three boxes sized by parameters; "17B active out of 400B" counter | Meta's newest Llama, its sizes, and the idea of *active* params |
| 1.2 | `act1/s02_big_picture.py` | tokens → embedding → stack of 48 blocks → norm → output; the residual stream as a horizontal "highway" | decoder-only transformer, residual stream |
| 1.3 | `act1/s03_block.py` | zoom into one block: RMSNorm → Attention → + ; RMSNorm → FFN/MoE → + | each block = attention + feed-forward, both added to the stream |
| 1.4 | `act1/s04_attention.py` | tokens with lines between them (causal triangle); 40 Q heads sharing 8 KV heads (GQA) | tokens look back at earlier tokens; GQA saves memory |
| 1.5 | `act1/s05_irope.py` | 4 layers: 3 have rotating arrows (RoPE) and chunk boxes, every 4th is plain NoPE with full-length lines | iRoPE: local position-aware layers + global layers → 10M context |
| 1.6 | `act1/s06_moe.py` | a router fanning tokens to 128 small expert boxes + 1 shared expert; only one lights up per token | Mixture of Experts: huge knowledge, small compute per token |

**Needs:** `BlockDiagram` (a reusable transformer block drawing), `TokenRow`, `ExpertGrid`, and an attention-lines helper.

## Act 2 — Our Mini Llama 4 (3 scenes)
Source: `mini_llama4/config.py`, `model.py`, `docs/llama4_architecture.md` §4.

| # | File | Visual | Narration (gist) |
|---|---|---|---|
| 2.1 | `act2/s01_shrink.py` | the Act 1 diagram shrinks; numbers morph: 5120→128, 48→4 layers, 128→4 experts, 202k→65 tokens, 400B→1.4M params | same design, laptop-sized |
| 2.2 | `act2/s02_layers.py` | the 4 mini layers labeled `dense/RoPE · MoE/RoPE · dense/RoPE · MoE/NoPE` | every key Llama 4 idea is still inside |
| 2.3 | `act2/s03_data.py` | TinyShakespeare text scrolls, characters turn into ids (real tokenizer) | what it learns from: a million characters of Shakespeare |

**Needs:** the same `BlockDiagram` as Act 1 (so the shrink is a direct morph), and real config values plus the vocabulary from `trace.json`.

## Act 3 — The code, line by line (9 scenes)
The code panel (left 40%) shows this hero snippet. It is a condensed version of `mini_llama4/train.py` and `model.py`, and every line is real:

```python
x, y   = batch[:, :-1], batch[:, 1:]          # 1
h      = model.tok_embeddings(x)              # 2
for layer in model.layers:                    # 3
    h = h + layer.attn(layer.attn_norm(h))    # 4
    h = h + layer.ffn(layer.ffn_norm(h))      # 5
logits = model.output(model.norm(h))          # 6
loss   = cross_entropy(logits, y)             # 7
loss.backward()                               # 8
opt.step(); opt.zero_grad()                   # 9
```

| # | File | Line | Visual (right 60%) | Narration (gist) |
|---|---|---|---|---|
| 3.1 | `act3/s01_targets.py` | 1 | token row and a shifted copy; arrows input → target | the target is the next token |
| 3.2 | `act3/s02_embed.py` | 2 | an id picks a row of the embedding table; the row flies out as a colored vector | tokens become vectors |
| 3.3 | `act3/s03_loop.py` | 3 | camera flies along the 4-layer stack from Act 2 | the vectors pass through 4 blocks |
| 3.4 | `act3/s04_attn.py` | 4 | RMSNorm squashes the vector; real attention weights as a heatmap (causal); the result is added to the stream | attention mixes in earlier tokens |
| 3.5 | `act3/s05_ffn.py` | 5 | router picks expert #k for each token (real routing); SwiGLU gate | each token gets its expert's processing |
| 3.6 | `act3/s06_logits.py` | 6 | vector → bar chart over 65 chars (real probabilities); the correct char is marked | a score for every possible next char |
| 3.7 | `act3/s07_loss.py` | 7 | correct bar → −log p → loss number (real ≈ 4.2) | loss = how surprised the model was |
| 3.8 | `act3/s08_backward.py` | 8 | **hero scene:** the graph glows in reverse from loss to the weights; red gradient arrows; chain rule labels on the edges | backprop: blame flows back through every line we just ran |
| 3.9 | `act3/s09_step.py` | 9 | a weight moves downhill on a loss curve; the loop spins; the real loss curve 4.2 → 1.8 draws | nudge and repeat: 300 steps later… |

**Needs:** `CodePanel` with `highlight_line(n)`, `TensorGrid` / `Heatmap` / `BarChart` fed by real data, and the `BlockDiagram` reused for the 4-layer stack.

## Outro (1 scene)
| `outro/s01_outro.py` | sample text before training (gibberish) vs. after (Shakespeare-ish, real samples); the cold-open loop returns | "Predict, measure, blame, nudge. That's how a Llama learns." |

---

## 3. What we need: shared foundation (built first, blocks every act)

| Item | File | Used by |
|---|---|---|
| Palette, fonts, timings | `video/common/style.py` | all |
| Narrated scene base (voice in one place) | `video/common/narration.py` | all |
| `TokenRow` (chars in boxes, with ids) | `video/common/tokens.py` | 0, 2, 3 |
| `BlockDiagram` (transformer block / stack, morphable) | `video/common/diagram.py` | 1, 2, 3 |
| `CodePanel` (hero snippet, `highlight_line`) | `video/common/code_panel.py` | 3 |
| `TensorGrid`, `Heatmap`, `BarChart`, `LossCurve` | `video/common/charts.py` | 3, outro |
| `TrainingLoop` diagram (predict→loss→backprop→update) | `video/common/loop.py` | 0, 3, outro |
| Real numbers: tokens, attention, routing, probs, loss, loss curve, samples | `video/make_trace.py` → `video/assets/trace.json`, read via `video/common/data.py` | 2, 3, outro |
| Render pipeline: render every scene in order, skip failures, concat → next `render-NN.mp4` | `video/render_all.py` | end |
| Stack test: 5 s, one narrated sentence | `video/test_clip.py` | pre-lab |

## 4. Agent split

| Agent | Owns | Starts when | Depends on |
|---|---|---|---|
| **FOUNDATION** | `video/common/*`, `make_trace.py`, `render_all.py`, `test_clip.py`, cold open 0.1 | first | — |
| **SCRIPT** (runs in parallel with FOUNDATION) | `SCRIPT.md`: final narration text and timing for every scene | first | the docs |
| **ACT 1** | `video/scenes/act1/*` | after `style`, `narration`, `diagram` exist | FOUNDATION, SCRIPT |
| **ACT 2** | `video/scenes/act2/*` | same | FOUNDATION (diagram must match Act 1), `trace.json` |
| **ACT 3** | `video/scenes/act3/*`, outro | after `code_panel` and `charts` exist | FOUNDATION, `trace.json` |
| **You (reviewer)** | watch `-ql` previews, write small fix tasks, send each to a free agent, merge | always | — |

Rules: an agent only edits files it owns. A missing shared helper is requested from FOUNDATION, not patched locally. Each agent uses its own branch or worktree (`act1`, `act2`, `act3`) and merges into `main` often.

**If time runs short, cut in this order:** Act 1 → scenes 1.1 and 1.5; Act 3 → merge 3.3 into 3.4. The cold open plus Act 3's lines 7–9 are the last things to cut, since they carry the takeaway.

## 5. Timeline
1. **Pre-lab:** install the stack, build `test_clip.py`, render `renders/test-01.mp4`, submit the quiz text.
2. **Phase 1:** FOUNDATION and SCRIPT run in parallel windows, and the cold open is rendered first.
3. **Phase 2:** ACT 1, ACT 2 and ACT 3 agents run in parallel, one scene per step, with a preview, a review and a commit each time.
4. **Phase 3 (last 20 min):** `render_all.py` → `/lab06-collect` → shared folder + Moodle.

## 6. Risks
- **manim-voiceover is incompatible with the installed Manim.** Fall back to `narration.py` making a gTTS mp3 and calling `add_sound` plus `wait`. Only one file changes.
- **gTTS is offline (no network).** Use `pyttsx3`, the Windows voice.
- **Act 1 and Act 2 diagrams drift apart.** Both must use `BlockDiagram` from common.
- **The video gets too long.** Trim the narration first, then follow the cut order above.
