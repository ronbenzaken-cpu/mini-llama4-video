# Video Plan: "How a Llama Learns: One Training Step, Line by Line"

## 1. Quiz answer (Week 6, last question)

| Field | Answer |
|---|---|
| **Topic** | How a large language model (Llama 4 architecture) is trained: the forward pass, the loss, backpropagation and the weight update |
| **After one minute the viewer understands that…** | *Training an LLM is one short loop repeated thousands of times: the model predicts the next token, the loss measures how wrong it was, backprop computes how much each weight caused the error, and the optimizer nudges every weight to be a little less wrong.* |
| **Stack** | Python 3.11 · Manim Community Edition (3Blue1Brown-style animation) · manim-voiceover with gTTS (narration) · PyTorch (real numbers from our mini Llama 4) · Claude Code · git/GitHub |

## 2. Constraints from the lab brief
- The video is **about 1 minute**. Each scene runs 5–12 s, which allows about 9 scenes.
- It needs **narration** throughout (TTS voice).
- **Every illustration is drawn in code** with Manim shapes and text. **No AI-generated images.**
- **Each scene lives in its own file** so several agents can work in parallel.
- **Small steps.** One scene, or one change to a scene, per step. Commit after each step with a *what + why* message.
- Use **low-quality previews** (`-ql`) while building, and render the full video only near the end.
- **Never overwrite a render.** Files are named `renders/render-01.mp4`, `render-02.mp4`, … with auto-incremented numbers.
- `SCRIPT.md` sits at the repo root and holds what each scene shows and what the narration says.
- Keep `.claude/skills/lab06-collect/` **out of git**.

## 3. "One scene per line": the hero code
Scene-per-line has to fit into 60 seconds, so the code on screen is a **7-line training step**. It is a condensed version of `mini_llama4/train.py` and runs the same way:

```python
x, y   = batch[:, :-1], batch[:, 1:]     # 1  inputs & next-token targets
h      = model.embed(x)                  # 2  tokens -> vectors
h      = model.blocks(h)                 # 3  attention + MoE, x4 layers
logits = model.head(model.norm(h))       # 4  a score for every next token
loss   = cross_entropy(logits, y)        # 5  how wrong were we?
loss.backward()                          # 6  backprop: blame each weight
opt.step(); opt.zero_grad()              # 7  nudge weights, repeat
```

Every scene follows the same pattern: the **code panel** (left) highlights its line, then the right side **animates what happens inside** that line, with **real tensors/values** from our trained mini Llama 4.

## 4. Scene list (target ≈ 62 s)

| # | File | Line | Visual (all drawn in Manim) | Narration (draft) | Sec |
|---|---|---|---|---|---|
| 00 | `s00_intro.py` | — | The text "To be or not to ___" with the model as a box, a question mark on the next word. Title appears. | "How does a model like Llama learn to finish your sentence? Let's watch one training step, line by line." | 6 |
| 01 | `s01_batch.py` | 1 | A row of character tokens, then a copy shifted one step left; arrows pair each input with its target | "We take text and shift it by one: every token's target is simply the next token." | 6 |
| 02 | `s02_embed.py` | 2 | Each token id lights up a row in an embedding table, and the row flies out as a column of colored cells | "Each token becomes a vector, a list of numbers the network can work with." | 6 |
| 03 | `s03_blocks.py` | 3 | A stack of 4 blocks. Inside one: attention lines between tokens (causal triangle), then a router sends tokens to experts (MoE) | "Inside four Llama blocks, attention lets tokens look at earlier tokens, and a router sends each one to an expert." | 9 |
| 04 | `s04_logits.py` | 4 | The final vector turns into a bar chart over the vocabulary; the true next char is marked | "At the end, the model scores every possible next token." | 6 |
| 05 | `s05_loss.py` | 5 | The bar for the correct token grows into "−log p"; the loss number appears (real value ≈ 4.2 at the start) | "The loss is low when the correct token got high probability, and high when it didn't." | 7 |
| 06 | `s06_backward.py` | 6 | **Hero scene:** the computation graph from loss back to the weights glows in reverse. Gradient arrows grow on each weight and the chain rule label appears on the edges | "Backpropagation runs the chain rule backward, telling every weight how much it contributed to the error." | 10 |
| 07 | `s07_step.py` | 7 | A weight dot on a loss-landscape curve takes a step downhill; then the loop arrow spins and the real loss curve (4.2 → 1.8) draws | "The optimizer nudges each weight downhill, and we repeat, thousands of times." | 7 |
| 08 | `s08_outro.py` | — | Generated text before training (gibberish) next to after training (Shakespeare-ish), using real samples; then the takeaway sentence | "Predict, measure, blame, nudge. That loop is how a language model learns." | 5 |

## 5. Repo layout to add
```
SCRIPT.md                     # full script (scene → visual → narration), root of repo
video/
  common/
    style.py                  # 3b1b palette, fonts, timings: shared contract
    code_panel.py             # CodePanel: the 7-line code, highlight_line(n)
    narration.py              # VoiceoverScene base with gTTS service (one place to swap the voice)
    data.py                   # loads video/assets/trace.json (real tokens, probs, loss, samples)
  scenes/s00_intro.py … s08_outro.py   # one class per file: Scene00 … Scene08
  test_clip.py                # 5-second stack test with one narrated sentence
  make_trace.py               # runs mini_llama4 once and saves real numbers → video/assets/trace.json
  render_all.py               # renders all scenes, concatenates → renders/render-NN.mp4 (next free number)
renders/                      # gitignored, never overwritten
```
`.gitignore` additions: `media/`, `renders/`, `.claude/skills/lab06-collect/`.

## 6. Shared contract (built first, before any parallel work)
- **Style:** dark background `#1C1C1C`. Colors are BLUE for data, YELLOW for highlight, RED for loss/gradients, GREEN for the update. Font is the same everywhere.
- **Layout:** the code panel fills the left 40% (it's the same in every scene) and the animation fills the right 60%. Each scene starts with the panel already on screen and its line highlighted, which makes the cuts between scenes seamless.
- **Narration:** every scene subclasses `NarratedScene` and wraps its animations in `with self.voiceover(text=...)`, so animation timing follows the voice.
- **Data:** scenes get numbers **only** from `common/data.py`, never invented.
- **Ownership:** an agent edits only the scene files it owns. Changes to `common/` go through one agent or the human.

## 7. Work plan and agent split

### Before the lab (homework)
1. Install the stack in `.venv`: `pip install manim "manim-voiceover[gtts]"`. LaTeX isn't needed because we use `Text` and `Code`, not `MathTex`.
2. Write `video/test_clip.py`: a 5-second scene with a title and one narrated sentence. Render it with `manim -ql`, then copy the result to `renders/test-01.mp4`. **Show it to the instructor at the start of the lab.**
3. Submit the quiz answer from section 1.
4. Commit and push.

### Phase 1: in the lab, two windows at once
- **Window A (visual + stack):** `common/style.py`, `common/code_panel.py`, `common/narration.py`, and `s00_intro.py` as the reference scene. Render a preview.
- **Window B (script):** write `SCRIPT.md` from section 4 and refine the narration wording and timings. Also run `make_trace.py` to produce the real numbers.

### Phase 2: parallel scene agents (each in its own git worktree/branch, merged often)
| Agent | Scenes | Why grouped |
|---|---|---|
| **START** | 01, 02 | data and embeddings (simple and fast) |
| **MIDDLE** | 03, 04 | the transformer blocks and logits (the most drawing work) |
| **END** | 05, 06, 07, 08 | loss, **backprop**, update, outro (the core message) |
| **You (human)** | review | watch each `-ql` preview, write fixes as small tasks, send them to a free agent |

Each task is a single scene or a single fix, a few minutes of work, and ends with a preview render and a commit (`"s06: slow gradient arrows so narration finishes first"`).

### Phase 3: last 20 minutes
1. `python video/render_all.py` writes `renders/render-NN.mp4`. A partial video is fine, since missing scenes are skipped.
2. Commit. Put `SKILL.md` in `.claude/skills/lab06-collect/` without committing it, then run `/lab06-collect` in a fresh session. That produces `lab06-video.mp4` and `lab06-bundle.zip`.
3. Copy the video into the shared folder as `ron-ben-zaken.mp4`, and upload both files to **Lab 6 report** on Moodle.

## 8. Risks and fallbacks
- **manim-voiceover is incompatible with the installed Manim version.** Fall back to `narration.py` making a gTTS mp3 itself and calling `self.add_sound()` plus `self.wait(duration)`. Only one file changes.
- **gTTS needs internet.** If the lab network is down, use `pyttsx3`, the offline Windows voice.
- **The video runs over 60 s.** Shorten the narration first and cut scene 03's MoE part second.
- **A scene is broken at the end.** `render_all.py` skips scenes that fail to render, so there's always a video.
