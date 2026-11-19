# HANDOFF: state of the project and what's left (written 2026-11-19)

Give this file to a new Claude Code session. Then read `PLAN.md` (full history), `SCRIPT.md` (scene briefs) and `AGENT_GUIDE.md` (how scenes are built and criticized).

Repo: https://github.com/ronbenzaken-cpu/mini-llama4-video (local: this folder; `gh` is logged in as `ronbenzaken-cpu`).

---

## 1. What the project is
A 3Blue1Brown-style narrated Manim video, "How a Llama Learns", for Lab 6. It has 20 scenes:
- **0.1** an opening
- **1.1–1.6** the real Llama 4
- **2.1–2.3** our Mini Llama 4
- **3.1–3.9** the 9-line training step, one scene per line
- **4.1** an outro

Every scene is a standalone file in `video/scenes/`. They are joined into one video by `video/render_all.py`.

## 2. User decisions (must keep)
- **Clarity over length.** Video length doesn't matter.
- **Every scene is standalone:** it starts empty and ends with `fade_out_all()`. Scenes are joined only at assembly.
- **Voice:** Ryan (`en-GB-RyanNeural`, Edge TTS, +8%). Gaps between sentences are very short (`READ_PAUSE` 0.15 s). There is a 0.8 s lead-in at the start of each scene, filled by the title or heading animation, because players swallowed "Today" at shorter lead-ins.
- **Visuals follow the voice:** bookmarks fire on exact words (Edge word timings → manim-voiceover bookmarks).
- **Transitions:** scroll or morph. Never cross-fade two texts in the same place.
- **Resolution:** 1080p60 for every saved version.
- **Versions:** keep every version of each scene with `video/save_version.py`, which writes `renders/scenes/<scene>/<scene>-vNN.mp4` and never overwrites.
- **Main video:** after **every** new scene version, run `.venv/Scripts/python video/render_all.py`. It writes `renders/render-NN.mp4`. **Tell the user the path so they can critique it.**
- **Scene 0.1 was built together with the user and is approved.** It's the model for all other scenes. Don't change it without the user.
- **Subagents:** one **builder** per scene, then one **critic** subagent per scene that fixes clarity and the **transition from the previous scene**. Every subagent first reads `AGENT_GUIDE.md`.
- Llama is by **Meta** (the user once said Google; the narration says Meta).
- The user likes working step by step and reviewing results before scaling up.

## 3. Status per scene (at handoff time)
| Scene | File | Builder | Critic | Latest saved version |
|---|---|---|---|---|
| 0.1 opening | `s0_1_cold_open.py` | ✅ built with the user | (approved by the user) | v04 |
| 1.1 family | `s1_1_family.py` | ✅ `e68217d` | ⏳ was running (the file has uncommitted critic edits) | v03 |
| 1.2 big picture | `s1_2_big_picture.py` | ⏳ in progress (file exists, uncommitted) | ☐ | — |
| 1.3 block | `s1_3_block.py` | ⏳ in progress | ☐ | — |
| 1.4 attention | `s1_4_attention.py` | ⏳ in progress | ☐ | — |
| 1.5 iRoPE | `s1_5_irope.py` | ⏳ in progress | ☐ | — |
| 1.6 MoE | `s1_6_moe.py` | ✅ committed `64fcbf1` (version maybe not saved yet) | ☐ | — |
| 2.1 shrink | `s2_1_shrink.py` | ✅ `a582f3c` | ☐ (waits for 1.6) | v01 |
| 2.2 layers | `s2_2_layers.py` | ⏳ in progress | ☐ | — |
| 2.3 data | `s2_3_data.py` | ⏳ in progress | ☐ | — |
| 3.1 targets | `s3_1_targets.py` | ⏳ in progress | ☐ | — |
| 3.2 embed | `s3_2_embed.py` | ⏳ rendering | ☐ | — |
| 3.3 layer loop | `s3_3_loop.py` | ✅ `11fdcb6` | ☐ (waits for 3.2) | v01 |
| 3.4 attention line | `s3_4_attn.py` | ⏳ in progress | ☐ | — |
| 3.5 ffn / experts | `s3_5_ffn.py` | ⏳ in progress | ☐ | — |
| 3.6 logits | `s3_6_logits.py` | ✅ `5afcdcf` | ☐ (waits for 3.5) | v01 |
| 3.7 loss | `s3_7_loss.py` | ⏳ in progress | ☐ | — |
| 3.8 backprop (hero) | `s3_8_backward.py` | ⏳ in progress | ☐ | — |
| 3.9 update | `s3_9_step.py` | ⏳ in progress | ☐ | — |
| 4.1 outro | `s4_1_outro.py` | ⏳ in progress | ☐ | — |

Main video so far: see the newest `renders/render-NN.mp4` (its `.txt` lists the scene versions used).

> ⚠️ **Background subagents belong to the old session.** If that session is closed, any builder or critic still running stops. Work it had finished is already in git commits and `renders/scenes/`. **To find what really finished,** check `git log --oneline` (a "Scene X.Y: …" commit means the builder finished) and `renders/scenes/<scene>/` (saved versions). A scene file that is **untracked or modified with no commit** is an interrupted build: give it to a new builder, which may reuse the file.

## 4. What's left to do
1. **Finish the builders.** For every scene without a commit and a saved version, launch a builder subagent with the prompt template in §5.
2. **Run the critics.** Start the critic for scene N when builders N and N−1 are both done. Order: 0.1 → 1.1 → … → 3.9 → 4.1 (`AGENT_GUIDE.md` §6). Use the critic prompt template in §5.
3. **After every new scene version,** run `render_all.py` and give the user the new `renders/render-NN.mp4` path.
4. **User review.** The user critiques the main video, and each note goes to the agent of that scene as a small fix, which produces a new version and then a new main render.
5. **Commit and push.** Builders and critics commit locally but don't push, so push regularly (`git push`). Never `git add -A`, because the folder `lab06-upload/` is untracked and isn't ours. Leave it alone and ask the user about it.
6. **Update `PLAN.md`** section 4 (done steps table) as steps finish.
7. **Lab submission (Phase 3, last 20 minutes of the lab):**
   - Commit.
   - Put the course's `SKILL.md` in `.claude/skills/lab06-collect/` (it's gitignored, so don't commit it).
   - Run `/lab06-collect` in a fresh session. It produces `lab06-video.mp4` and `lab06-bundle.zip`.
   - Copy the video to the section's shared folder as `ron-ben-zaken.mp4`.
   - Upload both files to **Lab 6 report** on Moodle. (Optional, until Dec 3: upload a finished version to "Lab 6 finished video".)
8. **Pre-lab items not yet confirmed:**
   - Submit the Week 6 quiz answer (text in `PLAN.md` §2).
   - Show the test clip `renders/test-01.mp4` to the instructor. It was made with the old gTTS voice; re-render it with `manim -qh video/test_clip.py TestClip` if the user wants the new voice.

## 5. Prompt templates
**Builder:**
> You are the BUILDER subagent for scene **X.Y "<title>"** of a 3Blue1Brown-style Manim video. Repo root: `C:\Users\Ron Ben Zaken\OneDrive\Desktop\Semester A\Agentic_Programing\6_alpha` (Windows; Bash is Git Bash, quote paths with spaces; run from the repo root). Your file: `video/scenes/<file>.py`, class `Scene_X_Y` (it may exist from an interrupted build; reuse or rebuild it). FIRST read `AGENT_GUIDE.md` completely and follow it exactly. Your brief is scene X.Y in `SCRIPT.md`; also read the briefs of the previous and next scenes. Other agents work in the same repo in parallel: touch only your file, `git add` only your file, never push. Your final report must follow the guide's builder report format.

**Critic:**
> You are the CRITIC subagent for scene **X.Y** of a 3Blue1Brown-style Manim video. Repo root: (as above). Scene file: `video/scenes/<file>.py`. Previous scene: **W.Z**, `video/scenes/<prev>.py` (latest render in `renders/scenes/<prev>/`). Do not edit the previous scene. FIRST read `AGENT_GUIDE.md` completely and follow its Critic workflow (section 5) exactly: check the transition from the end of W.Z into the start of X.Y and the clarity of X.Y; fix only your scene file with small targeted edits; render with -qh, save a version, commit only that file (message starting "critic:"), never push. Final report per the guide's critic format.

## 6. Known issues and tips (from agent reports)
- **No LaTeX on this machine.** No `MathTex`, `Tex`, `Integer` or `DecimalNumber`. Use counters built from `always_redraw(Text)` with a `ValueTracker`.
- **Consolas is missing some glyphs:** ✓ ✗ ∝ ␣. Use `marks.py` for ✓/✗. `data.show_char(" ")` now returns "·".
- **`interpolate_color`** needs `ManimColor` objects, not hex strings.
- **`CodePanel` text is fairly small** (reported by 3.3). It could be enlarged in `video/common/code_panel.py` (`PANEL_WIDTH`), but that changes every Act 3 scene, so ask the user first.
- **Real numbers are in `video/assets/trace.json`** (regenerate with `video/make_trace.py`): untrained loss 4.284, p(target) 0.0121 → trained 0.744, params 1,395,072.
- **ffmpeg** is installed via winget and is on PATH. `render_all.py` uses it.
