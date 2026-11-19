"""Scene 1.1 - The Llama 4 family.

Goal: Llama 4 is huge, but each token only uses a small part of it.
  1. Three boxes, AREA proportional to total parameters: Scout 109B, Maverick 400B,
     Behemoth ~2T (grey, dashed, "not released").
  2. Maverick morphs into a big frame that fills with its 128 experts (PURPLE);
     a word comes in and lights ONE of them (YELLOW): "17B active per token".
Facts: docs/llama4_architecture.md, section 1 (hard-coded constants, as the brief allows).

Style follows s0_1_cold_open.py: lead-in heading, bookmarks, morphs (no cross-fades).

Preview:  .venv/Scripts/python -m manim -ql video/scenes/s1_1_family.py Scene_1_1
Final:    .venv/Scripts/python -m manim -qh video/scenes/s1_1_family.py Scene_1_1
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, DashedVMobject, FadeIn,  # noqa: E402
                   GrowArrow, GrowFromEdge, Indicate, LaggedStart, Rectangle, Square,
                   Transform, VGroup, Write, smooth)

from video.common import style  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402
from video.common.tokens import token_box  # noqa: E402

# total parameters in billions (docs/llama4_architecture.md, section 1)
SCOUT_B, MAVERICK_B, BEHEMOTH_B = 109, 400, 2000
ACTIVE_B = 17
N_EXPERTS = 128
ROWS, COLS = 8, 16
BASELINE_Y = -1.7                      # the three boxes stand on this line
UNIT = 4.4 / math.sqrt(BEHEMOTH_B)     # side length per sqrt(billion): area ~ parameters


class Scene_1_1(NarratedScene):
    # ---------- builders ----------
    def param_box(self, billions: float, color, dashed: bool = False):
        side = UNIT * math.sqrt(billions)
        sq = Square(side_length=side, color=color, stroke_width=3,
                    fill_color=color, fill_opacity=0.0 if dashed else 0.25)
        if dashed:
            fill = Square(side_length=side, stroke_width=0, fill_color=style.GREY_BOX, fill_opacity=0.7)
            return VGroup(fill, DashedVMobject(sq, num_dashes=48))
        return sq

    def caption(self, box, name: str, size_txt: str, color, extra: str | None = None) -> VGroup:
        lines = [self.text(name, size=style.BODY_SIZE, color=color),
                 self.text(size_txt, size=style.LABEL_SIZE, color=style.TEXT)]
        if extra:
            lines.append(self.text(extra, size=style.LABEL_SIZE, color=style.DIM))
        return VGroup(*lines).arrange(DOWN, buff=0.1).next_to(box, DOWN, buff=0.22)

    # ---------- scene ----------
    def construct(self):
        # ================= 1. The family, drawn to scale =================
        heading = self.heading("Llama 4: the family")

        scout = self.param_box(SCOUT_B, style.BLUE)
        maverick = self.param_box(MAVERICK_B, style.PURPLE)
        behemoth = self.param_box(BEHEMOTH_B, style.DIM, dashed=True)
        for box, x in ((scout, -4.9), (maverick, -2.4), (behemoth, 3.0)):
            box.move_to([x, 0, 0]).align_to([0, BASELINE_Y, 0], DOWN)

        scout_cap = self.caption(scout, "Scout", "109B", style.BLUE)
        mav_cap = self.caption(maverick, "Maverick", "400B", style.PURPLE)
        beh_cap = self.caption(behemoth, "Behemoth", "~2T", style.DIM, "not released")
        note = self.text("box area = total parameters", size=style.LABEL_SIZE, color=style.DIM)
        note.to_corner(DOWN + RIGHT, buff=0.45)

        # the heading animates during the lead-in, so the first word lands on a moving screen
        self.play(Write(heading), run_time=style.LEAD_IN)
        with self.voiceover(text='Llama 4 is Meta\'s newest open model, and it comes in '
                                 '<bookmark mark="three"/>three sizes, drawn here to scale.'):
            self.wait_until_bookmark("three")
            self.play(FadeIn(note, shift=UP * 0.15), run_time=0.6)
        self.read_pause()

        with self.voiceover(text='<bookmark mark="scout"/>Scout has 109 billion parameters. '
                                 '<bookmark mark="mav"/>Maverick has 400 billion.'):
            self.wait_until_bookmark("scout")
            self.play(GrowFromEdge(scout, DOWN), FadeIn(scout_cap, shift=UP * 0.2), run_time=0.8)
            self.wait_until_bookmark("mav")
            self.play(GrowFromEdge(maverick, DOWN), FadeIn(mav_cap, shift=UP * 0.2), run_time=0.9)
        self.read_pause()

        with self.voiceover(text='And the giant, <bookmark mark="beh"/>Behemoth, about two trillion, '
                                 '<bookmark mark="nr"/>was never released.'):
            self.wait_until_bookmark("beh")
            self.play(FadeIn(behemoth[0]), Create(behemoth[1]), run_time=1.0)
            self.play(FadeIn(beh_cap[:2], shift=UP * 0.2), run_time=0.5)
            self.wait_until_bookmark("nr")
            self.play(FadeIn(beh_cap[2], shift=UP * 0.2), run_time=0.5)
        self.read_pause(3)  # let the viewer compare the three sizes

        # ================= 2. Inside Maverick: 128 experts =================
        cell = 0.4
        frame = Rectangle(width=COLS * cell + 0.3, height=ROWS * cell + 0.3, color=style.PURPLE,
                          stroke_width=3, fill_color=style.PURPLE, fill_opacity=0.08)
        frame.move_to(DOWN * 0.15)
        experts = VGroup(*[
            Square(side_length=cell * 0.78, stroke_width=1.5, color=style.PURPLE,
                   fill_color=style.PURPLE, fill_opacity=0.35)
            for _ in range(N_EXPERTS)
        ]).arrange_in_grid(rows=ROWS, cols=COLS, buff=cell * 0.22).move_to(frame)

        new_heading = self.heading("Llama 4 Maverick", color=style.PURPLE)
        total_lbl = self.text("400B parameters", size=style.BODY_SIZE, color=style.TEXT)
        total_lbl.next_to(frame, UP, buff=0.25).align_to(frame, LEFT)
        exp_lbl = self.text("128 experts", size=style.BODY_SIZE, color=style.PURPLE)
        exp_lbl.next_to(frame, UP, buff=0.25).align_to(frame, RIGHT)

        others = VGroup(scout, scout_cap, behemoth, beh_cap)
        with self.voiceover(text='Let\'s <bookmark mark="zoom"/>zoom into Maverick. '
                                 'It stores <bookmark mark="400"/>400 billion parameters.'):
            self.wait_until_bookmark("zoom")
            # morph: the small Maverick box grows into the big frame; the others slide away
            # the old heading leaves upward, then the new one drops in (never both in one spot)
            self.play(Transform(maverick, frame),
                      heading.animate.shift(UP * 1.2).set_opacity(0),
                      mav_cap.animate.shift(DOWN * 0.6).set_opacity(0),
                      scout.animate.shift(LEFT * 6), scout_cap.animate.shift(LEFT * 6),
                      behemoth.animate.shift(RIGHT * 7), beh_cap.animate.shift(RIGHT * 7),
                      note.animate.shift(RIGHT * 7),
                      run_time=1.2, rate_func=smooth)
            self.remove(others, mav_cap, note, heading)
            self.play(FadeIn(new_heading, shift=DOWN * 0.4), run_time=0.5)
            self.wait_until_bookmark("400")
            self.play(FadeIn(total_lbl, shift=UP * 0.2), run_time=0.6)
        self.read_pause()

        with self.voiceover(text='Most of them live in <bookmark mark="exp"/>128 experts: '
                                 '<bookmark mark="spec"/>smaller networks inside the model.'):
            self.wait_until_bookmark("exp")
            self.play(LaggedStart(*[FadeIn(e, scale=0.5) for e in experts], lag_ratio=0.02),
                      FadeIn(exp_lbl, shift=UP * 0.2), run_time=1.6)
            self.wait_until_bookmark("spec")
            self.play(Indicate(exp_lbl, color=style.PURPLE, scale_factor=1.1), run_time=0.6)
        self.read_pause()

        # ================= 3. One word uses one expert: 17B active =================
        lit = experts[5 * COLS + 9]
        glow = lit.copy().set_fill(style.YELLOW, opacity=1.0).set_stroke(style.YELLOW, width=3)
        tok = token_box("word", color=style.BLUE).next_to(frame, LEFT, buff=0.35)
        tok.shift(DOWN * (tok.get_y() - lit.get_y()))
        tok_arrow = Arrow(tok.get_right(), lit.get_left(), buff=0.08, color=style.YELLOW,
                          stroke_width=4, max_tip_length_to_length_ratio=0.08)
        active_lbl = self.text(f"{ACTIVE_B}B active per word", size=style.BODY_SIZE, color=style.YELLOW)
        active_lbl.next_to(frame, DOWN, buff=0.3)
        foot = self.text("= attention + 1 shared expert + 1 of 128 experts",
                         size=style.LABEL_SIZE, color=style.DIM).next_to(active_lbl, DOWN, buff=0.15)

        with self.voiceover(text='But <bookmark mark="word"/>each word only goes '
                                 'to <bookmark mark="one"/>one of them, '
                                 'so it uses just <bookmark mark="17"/>17 billion parameters.'):
            self.wait_until_bookmark("word")
            self.play(FadeIn(tok, shift=RIGHT * 0.3), run_time=0.6)
            self.wait_until_bookmark("one")
            self.play(experts.animate.set_fill(opacity=0.12).set_stroke(opacity=0.45),
                      GrowArrow(tok_arrow), FadeIn(glow, scale=1.6), run_time=0.8)
            self.wait_until_bookmark("17")
            self.play(Write(active_lbl), run_time=0.8)
        self.read_pause()

        with self.voiceover(text='The other <bookmark mark="idle"/>127 experts sit idle. '
                                 'Huge knowledge, <bookmark mark="cheap"/>small cost per word.'):
            self.wait_until_bookmark("idle")
            self.play(LaggedStart(*[e.animate.set_stroke(style.DIM, opacity=0.45)
                                    for e in experts if e is not lit], lag_ratio=0.004),
                      run_time=0.8)
            self.wait_until_bookmark("cheap")
            self.play(FadeIn(foot, shift=UP * 0.1),
                      Indicate(glow, color=style.YELLOW, scale_factor=1.4), run_time=0.8)
        self.read_pause(3)  # let the viewer take in the finished picture

        self.fade_out_all()
