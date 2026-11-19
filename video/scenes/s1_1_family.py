"""Scene 1.1: The Llama 4 family.

Goal: Llama 4 is huge, but each token only uses a small part of it.
Box areas are proportional to total parameter count (Scout 109B, Maverick 400B,
Behemoth ~2T). Maverick then zooms in and fills with its 128 routed experts;
one lights up = "17B active per token". Facts: docs/llama4_architecture.md section 1.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import math
from manim import *
from video.common import style
from video.common.narration import NarratedScene

# total parameters in billions (docs/llama4_architecture.md, section 1)
SCOUT_B, MAVERICK_B, BEHEMOTH_B = 109, 400, 2000
N_EXPERTS = 128
BASELINE_Y = -2.0         # bottoms of the three boxes line up here
UNIT = 4.4 / math.sqrt(BEHEMOTH_B)   # side length per sqrt(billion params)


def param_box(billions, color, fill_opacity=0.25, dashed=False):
    side = UNIT * math.sqrt(billions)          # area ∝ parameter count
    box = Square(side_length=side, color=color, stroke_width=3,
                 fill_color=color, fill_opacity=fill_opacity)
    if dashed:
        box = DashedVMobject(box, num_dashes=40)
    return box


def label(text, color=style.TEXT, size=style.LABEL_SIZE):
    return Text(text, font=style.FONT, font_size=size, color=color)


class Scene_1_1(NarratedScene):
    def construct(self):
        title = Text("Llama 4", font=style.FONT, font_size=style.HEADING_SIZE,
                     color=style.TEXT).to_edge(UP, buff=0.4)

        scout = param_box(SCOUT_B, style.BLUE)
        maverick = param_box(MAVERICK_B, style.PURPLE)
        behemoth = param_box(BEHEMOTH_B, style.DIM, fill_opacity=0.0, dashed=True)
        behemoth_fill = Square(side_length=behemoth.width, stroke_width=0,
                               fill_color=style.GREY_BOX, fill_opacity=0.6)

        for box, x in ((scout, -5.0), (maverick, -2.6), (behemoth, 2.6)):
            box.move_to([x, 0, 0]).align_to([0, BASELINE_Y, 0], DOWN)
        behemoth_fill.move_to(behemoth)

        def caption(box, name, size_txt, color, extra=None):
            lines = [label(name, color, style.LABEL_SIZE + 2),
                     label(size_txt, style.TEXT)]
            if extra:
                lines.append(label(extra, style.DIM, style.LABEL_SIZE - 4))
            g = VGroup(*lines).arrange(DOWN, buff=0.12)
            return g.next_to(box, DOWN, buff=0.25)

        scout_cap = caption(scout, "Scout", "109B", style.BLUE)
        mav_cap = caption(maverick, "Maverick", "400B", style.PURPLE)
        beh_cap = caption(behemoth, "Behemoth", "~2T", style.DIM, "not released")
        note = label("box area ~ total parameters", style.DIM, style.LABEL_SIZE - 4)
        note.to_edge(DOWN, buff=0.25).to_edge(RIGHT, buff=0.4)

        # ---- beat 1: title ----
        with self.voiceover(text="Meta's newest Llama is Llama 4.") as tracker:
            self.play(Write(title), run_time=tracker.duration)
        self.wait(0.3)

        # ---- beat 2: the family, one box at a time, drawn to scale ----
        with self.voiceover(
            text="It comes in three sizes, drawn here to scale: Scout, Maverick, "
                 "and the giant Behemoth, which was never released."
        ) as tracker:
            t = tracker.duration
            self.play(FadeIn(note), run_time=0.15 * t)
            self.play(GrowFromEdge(scout, DOWN), FadeIn(scout_cap, shift=UP * 0.2),
                      run_time=0.2 * t)
            self.play(GrowFromEdge(maverick, DOWN), FadeIn(mav_cap, shift=UP * 0.2),
                      run_time=0.2 * t)
            self.play(FadeIn(behemoth_fill), Create(behemoth),
                      FadeIn(beh_cap, shift=UP * 0.2), run_time=0.35 * t)
        self.wait(1.0)

        # ---- beat 3: zoom into Maverick ----
        rows, cols = 8, 16
        cell = 0.36
        target = Rectangle(width=cols * cell + 0.3, height=rows * cell + 0.3,
                           color=style.PURPLE, stroke_width=3,
                           fill_color=style.PURPLE, fill_opacity=0.08)
        target.move_to([0, -0.15, 0])
        experts = VGroup(*[
            Square(side_length=cell * 0.78, stroke_width=1.5, color=style.PURPLE,
                   fill_color=style.PURPLE, fill_opacity=0.35)
            for _ in range(N_EXPERTS)
        ]).arrange_in_grid(rows=rows, cols=cols, buff=cell * 0.22).move_to(target)

        new_title = Text("Llama 4 Maverick", font=style.FONT,
                         font_size=style.HEADING_SIZE, color=style.PURPLE).move_to(title)
        total_a = label("400B parameters", style.TEXT, style.BODY_SIZE - 4)
        total_b = label("· 128 experts", style.TEXT, style.BODY_SIZE - 4)
        total_lbl = VGroup(total_a, total_b).arrange(RIGHT, buff=0.22)
        total_lbl.next_to(target, UP, buff=0.3)

        with self.voiceover(text="Maverick stores 400 billion parameters.") as tracker:
            t = tracker.duration
            self.play(
                FadeOut(scout), FadeOut(scout_cap), FadeOut(behemoth),
                FadeOut(behemoth_fill), FadeOut(beh_cap), FadeOut(mav_cap), FadeOut(note),
                ReplacementTransform(maverick, target),
                ReplacementTransform(title, new_title),
                run_time=0.6 * t,
            )
            self.play(FadeIn(total_a, shift=UP * 0.2), run_time=0.4 * t)
        self.wait(0.5)

        # ---- beat 4: the 128 experts ----
        with self.voiceover(text="They are split into 128 experts.") as tracker:
            t = tracker.duration
            self.play(
                LaggedStart(*[FadeIn(e, scale=0.5) for e in experts], lag_ratio=0.03),
                FadeIn(total_b, shift=UP * 0.2),
                run_time=t,
            )
        self.wait(0.6)

        # ---- beat 5: one expert lights up per token ----
        lit = experts[6 * cols + 5]
        glow = lit.copy().set_fill(style.YELLOW, opacity=1.0).set_stroke(style.YELLOW, width=3)
        active_lbl = label("17B active per token", style.YELLOW, style.BODY_SIZE)
        active_lbl.next_to(target, DOWN, buff=0.35)
        arrow = Arrow(active_lbl.get_top() + UP * 0.05, lit.get_bottom(),
                      color=style.YELLOW, stroke_width=3, buff=0.08,
                      max_tip_length_to_length_ratio=0.12)
        footnote = label("attention + 1 shared expert + 1 of 128 routed experts",
                         style.DIM, style.LABEL_SIZE - 4)
        footnote.next_to(active_lbl, DOWN, buff=0.2)

        with self.voiceover(
            text="But each word only uses 17 billion of them: "
                 "a router sends it to just one expert, and the rest sit idle."
        ) as tracker:
            t = tracker.duration
            self.play(
                experts.animate.set_fill(opacity=0.12).set_stroke(opacity=0.5),
                total_lbl.animate.set_opacity(0.6),
                run_time=0.2 * t,
            )
            self.play(FadeIn(glow, scale=1.6), GrowArrow(arrow),
                      Write(active_lbl), run_time=0.35 * t)
            self.play(FadeIn(footnote, shift=UP * 0.1), run_time=0.2 * t)
            self.play(Indicate(glow, color=style.YELLOW, scale_factor=1.4), run_time=0.25 * t)

        self.wait(1.5)
        self.fade_out_all()
