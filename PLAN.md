# PLAN: "How a Llama Learns" (Lab 6 video project)

This is the master plan: what the project is, every step done so far, and what is left.
- Scene-by-scene content is in **`SCRIPT.md`**.
- Architecture facts are in **`docs/llama4_architecture.md`**.
- Repo: https://github.com/ronbenzaken-cpu/mini-llama4-video

---

## 1. The project in one paragraph
This is a ~3-minute, 3Blue1Brown-style narrated video. It explains Meta's **Llama 4** network, introduces our **Mini Llama 4** (the same design shrunk to 1.4M parameters), and then walks through a **9-line training step, one scene per line**: targets, embedding, layers, attention, experts, logits, loss, **backprop**, update. All animations are drawn in code with Manim, and every number on screen comes from our real trained model.

## 2. Quiz answer (Week 6, last question)
| Field | Answer |
|---|---|
| **Topic** | How a large language model (Meta's Llama 4) is built and trained: architecture, then forward pass, loss, backpropagation and weight update in real code |
| **After one minute the viewer understands that…** | *An LLM like Llama is a stack of attention and expert layers that predicts the next token, and training is one loop repeated thousands of times: predict, measure the error with the loss, backprop to find each weight's blame, nudge the weights.* |
| **Stack** | Python 3.11 · Manim Community Edition · manim-voiceover + Microsoft Edge neural TTS (narration) · PyTorch · Claude Code · git/GitHub |

## 3. Lab requirements we must meet
- [ ] Topic, takeaway sentence and stack submitted in the Week 6 quiz
- [x] Stack set up on the laptop, and a **5 s test clip with one narrated sentence** rendered before the lab
- [ ] `SCRIPT.md` in the repo (draft done ✓)
- [ ] Each scene in its own file. Small steps, one commit per step with a *what + why* message
- [ ] Previews at `-ql`. Full render only near the end
- [ ] Renders never overwritten: `renders/render-01.mp4`, `render-02.mp4`, …
- [ ] Visuals drawn in code. **No AI-generated images**
- [ ] Last 20 min: commit, then put `SKILL.md` in `.claude/skills/lab06-collect/` (not committed, already in `.gitignore`), then run `/lab06-collect` in a fresh session, which produces `lab06-video.mp4` and `lab06-bundle.zip`
- [ ] Copy the video to the section's shared folder as `ron-ben-zaken.mp4`, and upload both files to **Lab 6 report** on Moodle
- [ ] (Optional, until Dec 3) upload the finished version to **Lab 6 finished video**

---

## 4. What has been done ✅

| # | Step | Result | Commit |
|---|---|---|---|
| 1 | Chose the topic and model | Train a Llama-family model and explain every line. Decided on a small from-scratch model, not fine-tuning | — |
| 2 | Found the latest public Llama | **Llama 4** (Scout / Maverick, April 2025). Meta's 2026 open models are branded "Muse", and "Llama 4.5" is unverified | — |
| 3 | Created the git repo | `git init`, `.gitignore` (venv, data, checkpoints, media, renders, collect skill), `README.md`, `requirements.txt` | `088af5a` |
| 4 | Wrote the architecture document | `docs/llama4_architecture.md`: family, RMSNorm, GQA, iRoPE, QK-norm, SwiGLU, MoE, early fusion, training recipe, and a real-vs-mini table | `088af5a` |
| 5 | Implemented Mini Llama 4 | `mini_llama4/`: config, model (RMSNorm, RoPE, QK-norm, chunked/NoPE attention, temperature tuning, GQA, SwiGLU, MoE with shared + top-1 sigmoid routing), data (TinyShakespeare, char tokenizer), train (AdamW, warmup + cosine, grad clipping, CSV log, checkpoint, sampling) | `088af5a` |
| 6 | Tested it | 6 unit tests pass (layer pattern, shapes, MoE routing, masks, RoPE norm, overfitting). Fixed a `view`→`reshape` bug they caught | `088af5a` |
| 7 | Trained it | 1.40M params, 300 steps on CPU in 73 s, **val loss 4.16 → 1.80**, Shakespeare-like samples | — |
| 8 | Published on GitHub | Installed the GitHub CLI, logged in as `ronbenzaken-cpu`, created the **public** repo, pushed `main` | — |
| 9 | First video plan | `VIDEO_PLAN.md`: a 1-minute, 9-scene version | `072d4ae` |
| 10 | Restructured into 3 acts | Llama 4 → Mini Llama 4 → code line by line, plus a cold open so the takeaway lands within the first minute | `10fbcb4` |
| 11 | First script | `SCRIPT.md` with start/end continuity between scenes | `d3f36b5` |
| 12 | **Script made standalone** | Each scene now starts empty and ends with a fade-out, so any scene can be given to one subagent. Joining happens only at the end. `VIDEO_PLAN.md` was merged into this `PLAN.md` | `725373e` |
| 13 | **Pre-lab stack ready** | Installed Manim 0.22.1, manim-voiceover 0.4.0 and gTTS. Wrote `video/common/style.py`, `narration.py` (`NarratedScene`) and `video/test_clip.py`. **`renders/test-01.mp4`: 5.1 s with narration** ✓ | `9b6d1cb` |
| 14 | Version keeping + clarity rule | `video/save_version.py` saves `renders/scenes/<scene>/<scene>-vNN.mp4` without overwriting. Clarity beats length (SCRIPT rules 7 and 9) | `47c6915`, `6bb7e9f` |
| 15 | **First 2 scenes by subagents** | 0.1 Cold open (22.1 s, v01) and 1.1 Llama 4 family (32.5 s, v01 to v02). Each agent read PLAN/SCRIPT first and owned one file | `1e3fbc8`, `d4fae73` |
| 16 | Shared fixes from agent reports | Per-scene voice cache (parallel renders corrupted it), `marks.py` (✓/✗ shapes), no-LaTeX notes (`Integer`, missing glyphs) | `444ed51`, `b30df36` |
| 17 | **New voice + base example scene** | Replaced the robotic gTTS with Microsoft Edge neural TTS (`edge_voice.py`, voice in `style.VOICE`, 5 samples in `renders/voice-samples/`). Added `video/scenes/s0_0_base_example.py`, the reference scene every agent copies (SCRIPT rule 0). Added `tokens.py` (baseline-safe words, token boxes) and layout constants in `style.py`. New opening for 0.1: "Today we'll learn…" plus a 3-part roadmap | `60934ec` |
| 18 | **Scene 0.1 built with the user (the model scene)** | Word-synced bookmarks (Edge word timings), scroll transitions, 1080p60, silence trimmed between sentences (0.15 s pauses), 0.8 s lead-in filled by the title animation so "Today" is never swallowed. v04 | `dba2583`, this commit |
| 19 | **Foundation for all scenes** | `make_trace.py` → `video/assets/trace.json` (real tokens, attention, routing, probabilities, loss 4.284, gradient norms, loss curve, samples), `common/data.py`, `common/code_panel.py` (9-line hero code, `focus(n)`), and **`AGENT_GUIDE.md`** (builder and critic workflows) | this commit |
| 20 | **All scene agents launched** | One builder subagent per scene (1.1–4.1), then one critic subagent per scene that fixes clarity and the transition from the previous scene | — |

---

## 5. What is left ⏳

### Step A: pre-lab homework (before the lab)
| Task | Output | Done when |
|---|---|---|
| A1. Install the stack in `.venv` | `pip install manim "manim-voiceover[gtts]"` and update `requirements.txt` | `manim --version` works |
| A2. Narration helper | `video/common/narration.py`: a `NarratedScene` base class | the voice plays in a test |
| A3. Test clip | `video/test_clip.py`: a 5 s title and one narrated sentence, copied to `renders/test-01.mp4` | the instructor can watch it |
| A4. Submit the quiz answer | the text in section 2 | submitted |

### Step B: foundation (one agent, before any scene work)
These are drawing tools only. They hold no state between scenes.
| File | Contents |
|---|---|
| `video/common/style.py` | background `#1C1C1C`, colors (BLUE data, YELLOW highlight, RED loss/gradients, GREEN update), font, text sizes, fade time 0.5 s |
| `video/common/narration.py` | `NarratedScene`: a voiceover service set up in one place, with a gTTS → pyttsx3 fallback |
| `video/common/tokens.py` | `TokenRow(text, ids=None)` |
| `video/common/code_panel.py` | `CodePanel(highlight=n)`: the 9-line snippet on the left 40% |
| `video/common/charts.py` | `Heatmap`, `BarChart`, `LossCurve`, `VectorColumn` |
| `video/common/data.py` + `video/make_trace.py` | runs `mini_llama4` once and saves `video/assets/trace.json`: config, vocab, ids for "To be or", embeddings, attention map, routing, untrained probabilities, initial loss, gradient norms, loss curve, untrained and trained samples |
| `video/render_all.py` | renders every scene in order, **skips scenes that fail**, concatenates them into the next free `renders/render-NN.mp4` |

### Step C: scenes (parallel subagents, one scene per task)
Every scene brief in `SCRIPT.md` is a complete task. Suggested split:

| Agent | Scenes | Count |
|---|---|---|
| ACT-0/1 | 0.1, 1.1 – 1.6 | 7 |
| ACT-2 | 2.1 – 2.3, 4.1 | 4 |
| ACT-3a | 3.1 – 3.5 | 5 |
| ACT-3b | 3.6 – 3.9 (incl. hero 3.8) | 4 |

The assignment can change freely, since the scenes don't depend on each other. If 3 agents are free, give 3 scenes. The **prompt for a subagent** is:
> "Build scene X.Y from SCRIPT.md. Follow the 'Rules every scene follows'. Only create/edit `video/scenes/sX_Y_*.py`. Render with `manim -ql`, fix until done, commit with a what+why message."

The human reviews each preview, writes small fix tasks, and gives them to any free agent.

### Step D: assembly
- `render_all.py` → `renders/render-01.mp4`, … Watch it, list the fixes, do another round of fixes, and render the next number.
- Optionally render the final version at high quality (`-qh`).

### Step E: collect and submit (last 20 minutes of the lab)
Follow the checklist in section 3.

---

## 6. Repo layout (target)
```
PLAN.md  SCRIPT.md  README.md  requirements.txt
docs/llama4_architecture.md
mini_llama4/  config.py model.py data.py train.py
tests/test_model.py
video/
  common/   style.py narration.py tokens.py code_panel.py charts.py data.py
  scenes/   s0_1_cold_open.py … s4_1_outro.py      (20 files, one scene each)
  assets/   trace.json
  make_trace.py  render_all.py  test_clip.py
renders/   (gitignored)  test-01.mp4  render-01.mp4 …
```

## 7. Risks and fallbacks
- **manim-voiceover is incompatible with the installed Manim.** `narration.py` makes the gTTS mp3 itself and calls `add_sound` plus `wait`. Only one file changes.
- **No internet in the lab.** Use `pyttsx3`, the offline Windows voice.
- **A scene is broken at the end.** `render_all.py` skips it, so there's still a video.
- **Short on time.** Cut in this order: 1.1 and 1.5, then 3.3, then 2.2. Keep 0.1, 3.7, 3.8 and 3.9 at all costs, since they carry the takeaway.
- **Length is not a constraint** (user decision): clarity comes first. Scenes may run longer than the rough targets.
