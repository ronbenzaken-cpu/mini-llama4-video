# SCRIPT: "How a Llama Learns"

Each scene is described by **START** (the first frame), **BEATS** (what happens, in order), **END** (the last frame) and **VOICE** (narration draft).

**Continuity rule:** a scene's START frame is exactly the previous scene's END frame. The END is the hand-off contract between agents: the objects named in **HANDS OFF** must be on screen, in that position, when the scene ends. The next scene recreates them in its first frame, so the cut is invisible.

**Screen layout**
- Acts 0–2: full screen, centered diagrams.
- Act 3: `CodePanel` on the left 40%, animation on the right 60%.

---

## ACT 0: COLD OPEN

### 0.1 Cold open (`act0/s01_cold_open.py`, ~10 s)
- **START:** black screen.
- **BEATS:**
  1. `TokenRow` "To be or not to ___" types in, centered.
  2. A rounded box labeled **LLM** appears under it, and an arrow goes from the blank into the box.
  3. The box guesses **"eat"** (red ✗).
  4. The `TrainingLoop` draws around the box: **Predict → Loss → Backprop → Update**. The arrow spins around it 3 times, and a counter runs `step 1 → 300`.
  5. The guess morphs to **"be"** (green ✓).
- **END:** the LLM box is centered, the token row is above it, and the title **"How a Llama Learns"** fades in on top. The loop fades out.
- **HANDS OFF:** the LLM box (center) and the title.
- **VOICE:** "A language model learns by guessing the next word, measuring how wrong it was, and nudging its weights, thousands of times. Let's see how."

---

## ACT 1: THE REAL LLAMA 4

### 1.1 The family (`act1/s01_family.py`, ~10 s)
- **START:** the LLM box at center, with the title.
- **BEATS:**
  1. The title fades and the box label changes to **"Llama 4"**.
  2. The box splits into three boxes sized by parameter count: **Scout 109B**, **Maverick 400B**, **Behemoth ~2T** (grey, "not released").
  3. Maverick fills with a grid of 128 small squares, then 1 square lights up, with the label **"17B active per token"**.
- **END:** Maverick alone at center as a plain box. Scout and Behemoth have faded.
- **HANDS OFF:** the Maverick box (center).
- **VOICE:** "Meta's newest Llama is Llama 4. Maverick stores 400 billion parameters, but each word only uses 17 billion of them."

### 1.2 The big picture (`act1/s02_big_picture.py`, ~10 s)
- **START:** the Maverick box at center.
- **BEATS:**
  1. The box stretches horizontally into the pipeline: `tokens → Embedding → [Block ×48] → RMSNorm → Output → next-token probabilities`.
  2. A bright **residual stream** line runs through all the blocks.
  3. A vector dot travels along it from left to right.
- **END:** the full pipeline spans the screen, and **one block glows yellow**.
- **HANDS OFF:** the pipeline, with the highlighted block.
- **VOICE:** "Text becomes tokens, tokens become vectors, and the vectors flow through 48 identical blocks along a residual stream."

### 1.3 One block (`act1/s03_block.py`, ~10 s)
- **START:** the pipeline with the yellow block.
- **BEATS:**
  1. The camera zooms into the yellow block, which expands into the `BlockDiagram`: `RMSNorm → Attention → (+)`, then `RMSNorm → FFN/MoE → (+)`, with skip lines along the residual.
  2. Each (+) pulses as the vector passes through it.
- **END:** the `BlockDiagram` fills the screen, with the **Attention box highlighted**.
- **HANDS OFF:** the `BlockDiagram` (Attention highlighted).
- **VOICE:** "Each block does two things: attention, then a feed-forward network, and each result is added back to the stream."

### 1.4 Attention + GQA (`act1/s04_attention.py`, ~10 s)
- **START:** the `BlockDiagram` with Attention highlighted.
- **BEATS:**
  1. The Attention box opens up, and the token row "to be or not to" appears inside it.
  2. Lines run from the last token to earlier tokens, with thickness showing the weight. A causal triangle shows that a token never looks ahead.
  3. A heads view follows: **40 query dots** and **8 key/value dots**, with every 5 query dots connected to 1 key/value dot (GQA).
  4. The camera pulls back out of the block and further, until 4 stacked blocks are visible.
- **END:** a close-up of **4 stacked blocks** (generic, unlabeled).
- **HANDS OFF:** the stack of 4 blocks.
- **VOICE:** "Attention lets each token look back at earlier ones. Groups of query heads share their keys and values, which saves memory."

### 1.5 iRoPE (`act1/s05_irope.py`, ~10 s)
- **START:** the stack of 4 blocks.
- **BEATS:**
  1. Blocks 1–3 get a rotating-arrow badge **RoPE** and short chunk brackets (local attention).
  2. Block 4 gets a **NoPE** badge, with long lines spanning the whole sequence (global attention).
  3. The pattern repeats ×12, the counter reads "48 layers", and the label reads **"10M-token context"**.
  4. The camera returns to the 4 blocks, and **block 2's FFN part glows**.
- **END:** the stack of 4 blocks with the RoPE/NoPE badges, and block 2's FFN highlighted.
- **HANDS OFF:** the 4-block stack (block 2's FFN highlighted).
- **VOICE:** "Three out of four layers know token positions and look nearby. Every fourth layer drops positions and looks at everything. That mix is what reaches ten-million-token contexts."

### 1.6 Mixture of Experts (`act1/s06_moe.py`, ~10 s)
- **START:** the 4-block stack with block 2's FFN highlighted.
- **BEATS:**
  1. The camera zooms into the FFN, which becomes a **Router**, a **Shared expert** and a grid of **128 experts**.
  2. Tokens enter one at a time. Each one goes to the shared expert plus one expert that lights up, and its gate value σ is shown.
  3. The counter reads "400B stored / 17B used".
  4. The camera zooms all the way out to the full pipeline from 1.2, with 4 labels: **5120 wide · 48 layers · 128 experts · 202k vocab**.
- **END:** the full Llama 4 pipeline with the 4 number labels.
- **HANDS OFF:** the pipeline and its 4 number labels.
- **VOICE:** "Instead of one big feed-forward network, a router sends each token to one of 128 experts. The model holds huge knowledge, but each token uses only a small part of it."

---

## ACT 2: OUR MINI LLAMA 4

### 2.1 Shrink (`act2/s01_shrink.py`, ~12 s)
- **START:** the full pipeline with its 4 number labels.
- **BEATS:**
  1. The diagram scales down. The numbers count down: 5120→**128**, 48→**4** layers, 128→**4** experts, 202k→**65** characters. A 5th label, 400B→**1.4M** params, appears.
  2. A laptop outline is drawn around the diagram, with the title "Mini Llama 4".
- **END:** the mini pipeline at center, with **4 blocks** visible inside the laptop outline.
- **HANDS OFF:** the mini pipeline (4 blocks).
- **VOICE:** "We built the same design, shrunk: 1.4 million parameters instead of 400 billion. Small enough to train on a laptop in a minute."

### 2.2 Same ideas inside (`act2/s02_layers.py`, ~12 s)
- **START:** the mini pipeline in the laptop.
- **BEATS:**
  1. The laptop fades. The 4 blocks rotate into a vertical stack and grow.
  2. The blocks get labels: `0 dense/RoPE · 1 MoE/RoPE · 2 dense/RoPE · 3 MoE/NoPE`.
  3. A checklist on the left ticks off one item at a time: RMSNorm ✓ GQA ✓ iRoPE ✓ QK-norm ✓ SwiGLU ✓ MoE ✓.
  4. The checklist fades and the stack moves to the right 60%.
- **END:** the vertical labeled stack on the right; the left side is empty.
- **HANDS OFF:** `MiniStack` (right side).
- **VOICE:** "Every key idea of Llama 4 is still inside: four layers, alternating dense and expert layers, with a no-position layer at the top."

### 2.3 The data (`act2/s03_data.py`, ~11 s)
- **START:** `MiniStack` on the right.
- **BEATS:**
  1. On the left, TinyShakespeare text scrolls ("ROMEO: …").
  2. The 65-character vocabulary appears as a strip.
  3. The slice **"To be or"** is highlighted, and each character pops out into a `TokenRow` box with its **real id** under it.
  4. The text and the stack fade, and the `TokenRow` slides to the center of the right 60%.
- **END:** the `TokenRow` "To be or" (8 chars, with ids) at right-center; the left side is empty.
- **HANDS OFF:** the `TokenRow` with ids.
- **VOICE:** "It learns from a million characters of Shakespeare. Each character gets a number, its token id."

---

## ACT 3: THE CODE, LINE BY LINE
`CodePanel` (left) shows the 9 lines and is on screen for the whole act. On every scene's first beat, the panel highlights that scene's line. Finished lines get a dim ✓.

### 3.1 Line 1: targets (`act3/s01_targets.py`, ~9 s)
- **START:** the `TokenRow` with ids on the right; the left is empty.
- **BEATS:**
  1. `CodePanel` slides in from the left and **line 1** highlights.
  2. The TokenRow duplicates, and the copy shifts one step left.
  3. Labels **x** (inputs) and **y** (targets) appear, with an arrow pairing each x[i] with y[i].
- **END:** the **x row** on top and the **y row** below it, slightly dim; line 1 has a ✓.
- **HANDS OFF:** the x row and the y row.
- **VOICE:** "Line one: the target for every token is just the next token."

### 3.2 Line 2: embedding (`act3/s02_embed.py`, ~9 s)
- **START:** the x row and the y row.
- **BEATS:**
  1. **Line 2** highlights. The y row parks in the bottom-right corner (it's needed again in 3.6).
  2. The embedding table appears as a colored 65 × 128 grid.
  3. Each id lights up its row, and the row flies out to become a vertical colored vector above its token.
- **END:** a row of 8 vertical vectors, labeled **h (8×128)**; y is parked at the bottom right.
- **HANDS OFF:** the h vectors and the parked y.
- **VOICE:** "Line two: each id picks a row from a learned table, so every character becomes a vector of 128 numbers."

### 3.3 Line 3: the layer loop (`act3/s03_loop.py`, ~7 s)
- **START:** the h vectors and the parked y.
- **BEATS:**
  1. **Line 3** highlights.
  2. `MiniStack` from 2.2 reappears, lying horizontally, and the h vectors enter layer 0. The counter reads "layer 1 / 4".
  3. The camera zooms into layer 0, which expands into a mini `BlockDiagram` with the **attention path** highlighted.
- **END:** the mini `BlockDiagram` with the attention path highlighted, the h vectors at its input, and y parked.
- **HANDS OFF:** the `BlockDiagram` (attention highlighted), h and y.
- **VOICE:** "Line three: the vectors pass through the four blocks, one after another."

### 3.4 Line 4: attention (`act3/s04_attn.py`, ~10 s)
- **START:** the `BlockDiagram` with attention highlighted, h at its input.
- **BEATS:**
  1. **Line 4** highlights.
  2. RMSNorm acts: the vectors' lengths even out.
  3. The **real attention heatmap** (8×8, lower triangle) appears.
  4. One heatmap row turns into lines from that token to the earlier tokens.
  5. The result flows into the (+) node, and the h vectors shift color slightly.
- **END:** the `BlockDiagram` with the **FFN path highlighted** and h updated.
- **HANDS OFF:** the `BlockDiagram` (FFN highlighted), h and y.
- **VOICE:** "Line four: normalize, then attend. Each token mixes in information from the tokens before it, and the result is added back."

### 3.5 Line 5: feed-forward / experts (`act3/s05_ffn.py`, ~10 s)
- **START:** the `BlockDiagram` with FFN highlighted.
- **BEATS:**
  1. **Line 5** highlights.
  2. The camera cuts to layer 1, which is MoE: a router, a shared expert and 4 experts.
  3. Each token's vector flies to its expert, following the **real routing** from the trace, and then the result is added back.
  4. Fast-forward: layers 2 and 3 flash by, with a **NoPE** badge flashing on layer 3. The counter reaches "4 / 4".
- **END:** the stack fades and the **final h vectors** sit on the right; y is parked.
- **HANDS OFF:** the final h and y.
- **VOICE:** "Line five: the feed-forward step. In expert layers, a router picks one expert per token. After four blocks, each vector has absorbed the context."

### 3.6 Line 6: logits (`act3/s06_logits.py`, ~9 s)
- **START:** the final h vectors and the parked y.
- **BEATS:**
  1. **Line 6** highlights.
  2. The last token's vector is normalized, multiplied by the output matrix, and becomes **65 bars**. Softmax turns them into probabilities, using the real (untrained) values, which are close to uniform.
  3. The target character flies up from the parked y and marks its bar in yellow.
- **END:** the bar chart with the correct bar highlighted.
- **HANDS OFF:** the bar chart (correct bar highlighted).
- **VOICE:** "Line six: the model scores all 65 possible next characters. Before training, it has no idea. Every bar is about the same."

### 3.7 Line 7: loss (`act3/s07_loss.py`, ~9 s)
- **START:** the bar chart with the correct bar highlighted.
- **BEATS:**
  1. **Line 7** highlights.
  2. The correct bar's p (≈0.016) lifts out and turns into −log p ≈ **4.15**.
  3. Averaged over all tokens, it becomes a big red **loss = 4.21** (the real value).
  4. The bars fade, and a **computation graph** builds across the right side: `Embed W → L0 → L1 → L2 → L3 → Output W → logits → loss`.
- **END:** the horizontal computation graph, with the red loss node at its right end.
- **HANDS OFF:** the computation graph.
- **VOICE:** "Line seven: the loss. Minus the log of the probability we gave the right answer. High loss means very surprised."

### 3.8 Line 8: backprop (`act3/s08_backward.py`, ~14 s): hero scene
- **START:** the computation graph with the red loss node.
- **BEATS:**
  1. **Line 8** highlights.
  2. A red glow travels from loss leftward, node by node.
  3. On the first edge, **∂L/∂logits = p − onehot** shows as mini bars.
  4. Each later edge gets a "× ∂" chain-rule label.
  5. At every weight node, a red **gradient arrow** grows, with its length set by the real gradient norm.
  6. The residual skip lines carry the glow straight back, labeled "gradient highway".
  7. The counter reads "1.4M gradients computed".
- **END:** the graph is dimmed and every weight node has a red arrow.
- **HANDS OFF:** the weight nodes with their gradient arrows.
- **VOICE:** "Line eight is where learning happens. Backpropagation runs the chain rule backward through every line we just executed, and tells each of the 1.4 million weights how much it is to blame."

### 3.9 Line 9: update and repeat (`act3/s09_step.py`, ~10 s)
- **START:** the weight nodes with gradient arrows.
- **BEATS:**
  1. **Line 9** highlights.
  2. The camera zooms to one weight, which becomes a dot on a 1D loss curve. The dot steps downhill, opposite its arrow (green).
  3. The arrows vanish (`zero_grad`).
  4. The **TrainingLoop** from 0.1 returns and spins.
  5. The **real loss curve** draws from 4.2 to 1.8 as the step counter runs to 300.
- **END:** the loss curve on the right, and the `CodePanel` with all 9 lines ✓.
- **HANDS OFF:** the loss curve and the `CodePanel`.
- **VOICE:** "Line nine: AdamW nudges every weight a little downhill. Then we repeat. Three hundred steps later, the loss has dropped from 4.2 to 1.8."

---

## OUTRO

### 4.1 Before vs. after (`outro/s01_outro.py`, ~10 s)
- **START:** the loss curve and the `CodePanel`.
- **BEATS:**
  1. The panel and the curve slide out.
  2. Two text cards appear: **step 0** (a real untrained sample, gibberish) and **step 300** (a real sample, Shakespeare-like).
  3. The `TrainingLoop` comes to the center with the words **Predict · Measure · Blame · Nudge**.
  4. The title appears.
- **END:** the title card "How a Llama Learns", then a fade to black.
- **VOICE:** "Predict, measure, blame, nudge. That loop, repeated, is how a Llama learns."

---

## Hand-off objects (FOUNDATION builds these so START/END match exactly)
| Object | First appears | Reused in |
|---|---|---|
| `TokenRow` | 0.1 | 1.4, 2.3, 3.1, 3.2 |
| LLM / model box | 0.1 | 1.1 |
| `Pipeline` (horizontal Llama diagram) | 1.2 | 1.6, 2.1 |
| `BlockDiagram` | 1.3 | 1.4, 3.3, 3.4 |
| `BlockStack` (4 blocks, badges) | 1.4 | 1.5, 2.1, 2.2 (`MiniStack`), 3.3, 3.5 |
| `ExpertGrid` | 1.6 | 3.5 (4 experts) |
| `CodePanel` | 3.1 | all of Act 3, the outro |
| `BarChart` | 3.6 | 3.7 |
| `CompGraph` | 3.7 | 3.8, 3.9 |
| `TrainingLoop` | 0.1 | 3.9, 4.1 |
| `LossCurve` | 3.9 | 4.1 |

Each hand-off object's **final position and scale** is a named constant in `video/common/layout.py`, for example `LAYOUT.tokenrow_right_center` or `LAYOUT.stack_right`. Both sides of a cut use the same constant.

## Real data needed in `trace.json`
Config numbers · the vocabulary · ids for "To be or" · an 8×8 attention map (layer 0, head 0) · layer-1 expert routing for the 8 tokens · untrained probabilities over the 65 characters for the last token · initial loss · per-node gradient norms · the loss curve CSV · an untrained sample text · a trained sample text.

**Total ≈ 3 min 10 s**
