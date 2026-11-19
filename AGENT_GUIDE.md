# AGENT GUIDE: how to build (and criticize) one scene

You are one of many subagents. Each scene has **two** subagents:
- a **BUILDER**, who makes the scene, and
- a **CRITIC**, who runs after it. The critic checks the scene and makes the **transition from the previous scene** feel natural.

Read this file fully before doing anything.

---

## 1. Read first (in this order)
1. `PLAN.md`, sections 1 and 5: what the project is.
2. `SCRIPT.md`, the rules at the top and **your scene's brief**. Also read the briefs of the scenes **before and after** yours.
3. **`video/scenes/s0_1_cold_open.py`**: the approved model scene, built with the user. Copy its techniques.
4. `video/scenes/s0_0_base_example.py`: annotated conventions.
5. `video/common/*`: `style.py`, `narration.py`, `tokens.py`, `marks.py`, `code_panel.py`, `data.py`.

## 2. What you own
- **Builder:** exactly one file, `video/scenes/s<act>_<n>_<name>.py`, with class `Scene_<act>_<n>`. You may not edit any other file.
- **Critic:** the same file, plus edits to it.
- Do not edit `video/common/*`, `SCRIPT.md`, `PLAN.md` or other scenes. If something shared is missing, write it inside your own file and mention it in your report.

## 3. The approved style (learned with the user on scene 0.1)
| Topic | Rule |
|---|---|
| **Explain first** | Open by saying **what this scene is about** in plain words ("Now let's look inside one block."). The user rejected scenes that start without context. |
| **Voice** | Ryan (Edge neural TTS), set in `style.py`. Never change the voice, speed or pauses in your file. |
| **Voice ↔ animation** | Use **bookmarks** so every visual change happens on the word that names it: `text='...when it <bookmark mark="p"/>predicts...'` then `self.wait_until_bookmark("p")` then `self.play(...)`. Don't hard-code `t.duration * 0.4` timing. |
| **Lead-in** | The scene's first animation (heading / title) plays **before** the first `voiceover` and lasts `style.LEAD_IN` (see 0.1). This avoids silence and stops players from clipping the first word. |
| **Transitions** | Inside a scene, no "fade everything out, fade new things in". Use **morphs** (`Transform`, `.animate`) or a **scroll** (old content slides up off screen while the new heading slides in from below: `SCROLL = UP*8`, see 0.1). Never cross-fade two texts in the same spot. |
| **Pauses** | `self.read_pause()` after each narration block. It is deliberately tiny (0.15 s). Use `read_pause(3-4)` only where the viewer needs to look at a finished picture. |
| **One idea at a time** | Each narration sentence introduces at most one new visual idea. Clarity beats length, and length doesn't matter. |
| **Colors by meaning** | BLUE data/model · YELLOW focus · RED loss/errors/gradients · GREEN correct/updates · PURPLE experts · DIM side notes. |
| **Text** | `self.text()`, `self.heading()`, and `tokens.word()` / `token_box()` / `token_row()` for tokens (baseline-safe). The minimum readable size is `style.LABEL_SIZE`. Keep everything inside the frame with a margin. |
| **No LaTeX** | No `MathTex`, `Tex`, `Integer` or `DecimalNumber`. Counters are `always_redraw(Text)` driven by a `ValueTracker`. ✓/✗ come from `marks.py`. Consolas lacks some symbols; check them in the stills. |
| **Real numbers** | All numbers about our mini model come from `video.common.data.TRACE`. **The numbers in SCRIPT.md are approximate. TRACE wins** (for example, the untrained loss is 4.284 and p(target) is 0.0121). Llama 4 facts come from `docs/llama4_architecture.md`. |
| **Act 3 layout** | `CodePanel()` on the left (`code_panel.py`). Call `panel.focus(n)` for the scene's line, on the bookmark where that line is spoken. The animation goes in the right area, around `style.ANIM_CENTER`, about 7 units wide. |
| **Ending** | `self.fade_out_all()`. Optionally end on a one-line teaser of the next scene's topic. |

## 4. Builder workflow
1. Write the scene, starting from the structure of `s0_1_cold_open.py`.
2. Preview: `.venv/Scripts/python -m manim -ql video/scenes/<file>.py Scene_X_Y`
3. Check stills. Extract about 8 frames into a contact sheet and look at it with the Read tool:
   ```
   .venv/Scripts/python -c "import av,sys;from PIL import Image;c=av.open(sys.argv[1]);fr=[f.to_image() for f in c.decode(video=0)];n=len(fr);ims=[fr[int(n*k/8)] for k in range(8)];w,h=ims[0].size;g=Image.new('RGB',(w*4,h*2));[g.paste(im,((i%4)*w,(i//4)*h)) for i,im in enumerate(ims)];g.save(sys.argv[2])" <mp4> <out.png>
   ```
   Fix any overlap, off-screen text, unreadable size, missing glyph or empty stretch.
4. Final render at **1080p60**: `.venv/Scripts/python -m manim -qh video/scenes/<file>.py Scene_X_Y`
5. Save the version: `.venv/Scripts/python video/save_version.py video/scenes/<file>.py Scene_X_Y -m "what changed"`
6. Commit: `git add <your file>` only (never `-A`). Write a what + why message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. If `index.lock` exists, wait and retry. **Do not push.**
7. Report: file, commit, version path, duration, beat list (bookmark → visual), and any problems.

## 5. Critic workflow (runs after the builder)
Your job is to make the scene **clear on its own** and make it **join smoothly with the previous scene**.
1. Read sections 1–3. Then read the **previous scene's file** and look at its **last 3 seconds** (stills). Read **your scene's file** and look at its **first 5 seconds**. If the previous scene isn't rendered yet, render it yourself at `-ql` (don't save a version or commit it).
2. Check the **transition**, from the end of the previous scene to the start of this one:
   - Does the opening say what this scene is about, and **connect to what was just shown**? ("We saw that Llama is built from blocks. Now let's look inside one.")
   - No repeated explanation of something the previous scene already explained, and no jump to a topic that hasn't been introduced.
   - Same visual language: heading position, the same colors for the same concepts, the same names (for example "residual stream", not "highway" in one scene and "stream" in the next).
   - The first visual matches the first words, and the lead-in is applied.
3. Check **clarity** inside the scene: one idea per sentence, bookmark sync, no cross-fades, nothing unreadable, every number taken from TRACE or the doc.
4. **Fix** what you find in the scene file. Make small, targeted edits; do not rewrite the scene. Then repeat steps 2–6 of the builder workflow, with a message like `critic: connect opening to scene 1.2's block diagram`.
5. Report: what you changed and why, the new version path, and anything you could not fix.

## 6. Scene order (for "previous scene")
0.1 → 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 2.1 → 2.2 → 2.3 → 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7 → 3.8 → 3.9 → 4.1
