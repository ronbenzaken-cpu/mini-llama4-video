# SCRIPT: "How a Llama Learns"

Each scene is a **standalone brief**. Give one brief to one subagent, and it can build that scene from first frame to last without knowing anything about the other scenes. The scenes are joined only at the end, by `video/render_all.py`.

## Rules every scene follows (the only shared contract)
1. **Starts on an empty dark background** and builds everything it needs itself, with nothing carried over from another scene.
2. **Ends by fading everything out** to the empty background (0.5 s). Because of this, any two scenes can be glued together in any order.
3. Uses only `video/common/style.py` (colors, fonts, sizes) and `video/common/narration.py` (the voice). It may also use the small helpers in `video/common/` (`TokenRow`, `CodePanel`, charts) as drawing tools. They don't hold state between scenes.
4. Real numbers come only from `video/common/data.py` (which reads `video/assets/trace.json`). Nothing is invented.
5. Lives in its own file, `video/scenes/<id>_<name>.py`, with one class named `Scene_<id>` (for example `Scene_3_8`).
6. All illustrations are drawn in code. No AI-generated images.
7. **Done when:** `manim -ql` renders without errors, the narration finishes before the fade-out, the length is within ±3 s of the target, and the scene has been committed.

**Act 3 layout:** `CodePanel` on the left 40% showing the 9-line snippet, with the scene's line highlighted. The animation fills the right 60%.

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

---

## ACT 0: COLD OPEN

### 0.1 Cold open · `s0_1_cold_open.py` · ~10 s
- **Goal:** state the whole idea of the video in one shot.
- **Shows:**
  1. "To be or not to ___" types in, with an **LLM** box below it.
  2. The box guesses "eat" (red ✗).
  3. A loop diagram **Predict → Loss → Backprop → Update** draws around the box and spins three times while a counter runs to 300.
  4. The guess becomes "be" (green ✓).
  5. The title "How a Llama Learns" appears.
- **Voice:** "A language model learns by guessing the next word, measuring how wrong it was, and nudging its weights, thousands of times. Let's see how."
- **Data:** none.

---

## ACT 1: THE REAL LLAMA 4
Facts come from `docs/llama4_architecture.md`.

### 1.1 The Llama 4 family · `s1_1_family.py` · ~10 s
- **Goal:** Llama 4 is huge, but each token only uses a small part of it.
- **Shows:**
  1. Three boxes sized by parameter count: Scout 109B, Maverick 400B, Behemoth ~2T (grey, "not released").
  2. Maverick fills with 128 small squares, and one lights up with the label "17B active per token".
- **Voice:** "Meta's newest Llama is Llama 4. Maverick stores 400 billion parameters, but each word only uses 17 billion of them."
- **Data:** numbers from the doc (hard-coded constants are fine).

### 1.2 The big picture · `s1_2_big_picture.py` · ~10 s
- **Goal:** an LLM is a pipeline: tokens → vectors → a stack of blocks → next-token probabilities.
- **Shows:**
  1. The pipeline is drawn left to right: `tokens → Embedding → [Block ×48] → RMSNorm → Output → probabilities`.
  2. A bright residual-stream line runs through the blocks, and a dot travels along it.
- **Voice:** "Text becomes tokens, tokens become vectors, and the vectors flow through 48 identical blocks along a residual stream."

### 1.3 Inside one block · `s1_3_block.py` · ~10 s
- **Goal:** a block is attention plus a feed-forward network, and each result is *added* to the stream.
- **Shows:**
  1. The block diagram: `RMSNorm → Attention → (+)`, then `RMSNorm → FFN/MoE → (+)`, with skip lines.
  2. A vector passes through, and each (+) pulses.
- **Voice:** "Each block does two things: attention, then a feed-forward network, and each result is added back to the stream."

### 1.4 Attention and GQA · `s1_4_attention.py` · ~10 s
- **Goal:** tokens look back at earlier tokens; query heads share key/value heads.
- **Shows:**
  1. Tokens "to be or not to". Lines run from the last token to earlier ones, with thickness showing weight. A causal triangle shows that no token looks ahead.
  2. 40 query dots and 8 KV dots, with every 5 query dots linked to 1 KV dot.
- **Voice:** "Attention lets each token look back at earlier ones. Groups of query heads share their keys and values, which saves memory."

### 1.5 iRoPE · `s1_5_irope.py` · ~10 s
- **Goal:** 3 of every 4 layers are local and know positions; every 4th is global with no positions.
- **Shows:**
  1. A stack of 4 blocks. Blocks 1–3 get a rotating-arrow "RoPE" badge and short chunk brackets. Block 4 gets a "NoPE" badge and long full-sequence lines.
  2. The pattern repeats ×12, the counter reads "48 layers", and the label reads "10M-token context".
- **Voice:** "Three out of four layers know token positions and look nearby. Every fourth layer drops positions and looks at everything. That mix reaches ten-million-token contexts."

### 1.6 Mixture of Experts · `s1_6_moe.py` · ~10 s
- **Goal:** a router sends each token to 1 of 128 experts plus a shared expert.
- **Shows:**
  1. A router box, a shared expert, and a grid of 128 experts.
  2. Tokens enter one at a time. Each lights up the shared expert plus one routed expert, with its gate value σ shown.
  3. The counter reads "400B stored / 17B used".
- **Voice:** "Instead of one big feed-forward network, a router sends each token to one of 128 experts. Huge knowledge, small cost per token."

---

## ACT 2: OUR MINI LLAMA 4

### 2.1 Shrinking it · `s2_1_shrink.py` · ~12 s
- **Goal:** same design, laptop-sized numbers.
- **Shows:**
  1. A small pipeline diagram with five labels whose numbers count down: 5120→128 width, 48→4 layers, 128→4 experts, 202k→65 vocab, 400B→1.4M params.
  2. A laptop outline is drawn around it, with the title "Mini Llama 4".
- **Voice:** "We built the same design, shrunk: 1.4 million parameters instead of 400 billion. Small enough to train on a laptop."
- **Data:** config values and the parameter count (`data.py`).

### 2.2 Same ideas inside · `s2_2_layers.py` · ~12 s
- **Goal:** every Llama 4 idea survives in the mini model.
- **Shows:**
  1. A vertical stack of 4 blocks labeled `0 dense/RoPE · 1 MoE/RoPE · 2 dense/RoPE · 3 MoE/NoPE`.
  2. A checklist ticks off one item at a time: RMSNorm ✓ GQA ✓ iRoPE ✓ QK-norm ✓ SwiGLU ✓ MoE ✓.
- **Voice:** "Every key idea of Llama 4 is still inside: four layers, alternating dense and expert layers, with a no-position layer at the top."
- **Data:** the layer pattern (`data.py`).

### 2.3 The data · `s2_3_data.py` · ~11 s
- **Goal:** the model learns from Shakespeare, one character = one token id.
- **Shows:**
  1. Shakespeare text scrolls.
  2. The 65-character vocabulary appears as a strip.
  3. "To be or" is highlighted, and each character pops into a box with its real id underneath.
- **Voice:** "It learns from a million characters of Shakespeare. Each character gets a number, its token id."
- **Data:** a text excerpt, the vocabulary, the ids for "To be or".

---

## ACT 3: THE CODE, LINE BY LINE
Every scene draws its own `CodePanel` (left), highlights its line, and animates on the right.

### 3.1 Line 1: targets · `s3_1_targets.py` · ~9 s
- **Goal:** the target for each token is the next token.
- **Shows:**
  1. The token row "To be or" (with ids) and a copy shifted one step left.
  2. Labels x and y, with arrows pairing each x[i] with y[i].
- **Voice:** "Line one: the target for every token is just the next token."
- **Data:** the ids.

### 3.2 Line 2: embedding · `s3_2_embed.py` · ~9 s
- **Goal:** each id selects a learned row and becomes a vector.
- **Shows:**
  1. A token row and an embedding table drawn as a colored grid (65 rows; draw about 12 visible rows with "⋮").
  2. Each id lights up its row, and the row flies out to become a vertical colored vector above its token, labeled "h: 8 × 128".
- **Voice:** "Line two: each id picks a row from a learned table, so every character becomes a vector of 128 numbers."
- **Data:** the ids and the real embedding values (first 8 dims are enough for color).

### 3.3 Line 3: the layer loop · `s3_3_loop.py` · ~7 s
- **Goal:** the vectors pass through the 4 blocks in order.
- **Shows:**
  1. A horizontal stack of 4 labeled blocks.
  2. The row of vectors enters block 0, comes out, enters block 1, and so on, while a counter runs 1/4 → 4/4.
- **Voice:** "Line three: the vectors pass through the four blocks, one after another."

### 3.4 Line 4: attention · `s3_4_attn.py` · ~10 s
- **Goal:** normalize, attend to earlier tokens, add back to the stream.
- **Shows:**
  1. Vectors of uneven length become even (RMSNorm).
  2. The real 8×8 attention heatmap appears, lower triangle only.
  3. One row of the heatmap becomes lines to the earlier tokens.
  4. The result passes through a (+) and the vectors change color slightly.
- **Voice:** "Line four: normalize, then attend. Each token mixes in information from the tokens before it, and the result is added back."
- **Data:** the attention map (layer 0, head 0) and the tokens.

### 3.5 Line 5: feed-forward / experts · `s3_5_ffn.py` · ~10 s
- **Goal:** in an MoE layer the router picks one expert per token, plus the shared expert.
- **Shows:**
  1. A router, a shared expert and 4 experts.
  2. Each of the 8 tokens flies to its real chosen expert (colored by expert), then the results are added back.
  3. A small "NoPE" badge note mentions that layer 3 has no positions.
- **Voice:** "Line five: the feed-forward step. In expert layers, a router picks one expert per token. After four blocks, each vector has absorbed its context."
- **Data:** the layer-1 routing for the 8 tokens.

### 3.6 Line 6: logits · `s3_6_logits.py` · ~9 s
- **Goal:** the model scores all 65 possible next characters.
- **Shows:**
  1. One vector is multiplied by the output matrix and becomes 65 bars, which softmax turns into probabilities. These are the real untrained values, so the bars are nearly flat.
  2. The correct next character's bar is marked in yellow.
- **Voice:** "Line six: the model scores all 65 possible next characters. Before training, it has no idea. Every bar is about the same."
- **Data:** the untrained probabilities and the target id.

### 3.7 Line 7: loss · `s3_7_loss.py` · ~9 s
- **Goal:** loss = −log(probability of the right answer), averaged.
- **Shows:**
  1. The correct bar's p (≈0.016) lifts out and turns into −log p ≈ 4.15.
  2. Averaged over the tokens, this becomes a big red "loss = 4.21".
  3. A small side graph of −log p shows that high p means low loss.
- **Voice:** "Line seven: the loss. Minus the log of the probability we gave the right answer. High loss means very surprised."
- **Data:** the p of the target and the initial loss.

### 3.8 Line 8: backprop (hero scene) · `s3_8_backward.py` · ~14 s
- **Goal:** backprop runs the chain rule backward and gives every weight a gradient.
- **Shows:**
  1. A computation graph drawn left to right: `Embed W → L0 → L1 → L2 → L3 → Output W → logits → loss`.
  2. A red glow travels from the loss leftward, node by node.
  3. The first edge shows ∂L/∂logits = p − onehot as mini bars, and each later edge gets a "× ∂" chain label.
  4. A red gradient arrow grows on every weight node, its length set by the real gradient norm. The residual skip lines carry the glow straight back, labeled "gradient highway".
  5. The counter reads "1.4M gradients".
- **Voice:** "Line eight is where learning happens. Backpropagation runs the chain rule backward through every line we just executed, and tells each of the 1.4 million weights how much it is to blame."
- **Data:** the per-module gradient norms.

### 3.9 Line 9: update and repeat · `s3_9_step.py` · ~10 s
- **Goal:** AdamW nudges each weight downhill; repeating this lowers the loss.
- **Shows:**
  1. A dot on a 1D loss curve with a red gradient arrow steps downhill (green), and then the arrow vanishes (zero_grad).
  2. A loop icon spins while the real loss curve draws from 4.2 to 1.8 and a step counter runs to 300.
- **Voice:** "Line nine: AdamW nudges every weight a little downhill. Then we repeat. Three hundred steps later, the loss has dropped from 4.2 to 1.8."
- **Data:** the loss curve.

---

## OUTRO

### 4.1 Before vs. after · `s4_1_outro.py` · ~10 s
- **Goal:** show the result, then repeat the takeaway.
- **Shows:**
  1. Two text cards: **step 0** (a real untrained sample, gibberish) and **step 300** (a real trained sample).
  2. The loop diagram with the words **Predict · Measure · Blame · Nudge**.
  3. The title.
- **Voice:** "Predict, measure, blame, nudge. That loop, repeated, is how a Llama learns."
- **Data:** the two samples.

---
**Total ≈ 3 min 10 s · 20 scenes**
