"""Scene 0.1 - Cold open: the whole idea of the video in one shot.

A sentence with a blank, an LLM box that guesses wrong, a training loop
(Predict -> Loss -> Backprop -> Update) that spins while a step counter runs
to 300, then the right guess and the title.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
from manim import *
from video.common import style
from video.common.narration import NarratedScene


def T(s, size=style.BODY_SIZE, color=style.TEXT, **kw):
    return Text(s, font=style.FONT, font_size=size, color=color, **kw)


def cross_mark(color=style.RED, size=0.42):
    """A drawn ✗ (the Consolas font has no glyph for it)."""
    h = size / 2
    return VGroup(Line([-h, -h, 0], [h, h, 0]), Line([-h, h, 0], [h, -h, 0])).set_stroke(color, 7)


def check_mark(color=style.GREEN, size=0.42):
    """A drawn ✓."""
    h = size / 2
    return VMobject(fill_opacity=0).set_points_as_corners(
        [[-h, 0, 0], [-h * 0.25, -h * 0.8, 0], [h, h, 0]]).set_stroke(color, 7)


class Scene_0_1(NarratedScene):
    def construct(self):
        # ---- sentence with a blank ----
        prompt = T("To be or not to", size=style.HEADING_SIZE)
        blank = T("___", size=style.HEADING_SIZE, color=style.DIM)
        sentence = VGroup(prompt, blank).arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
        sentence.move_to(UP * 2.9)

        # ---- the LLM box ----
        center = DOWN * 0.3
        box = RoundedRectangle(width=2.2, height=1.2, corner_radius=0.15,
                               stroke_color=style.BLUE, stroke_width=4,
                               fill_color=style.GREY_BOX, fill_opacity=1).move_to(center)
        box_label = T("LLM", size=style.HEADING_SIZE, color=style.BLUE).move_to(box)
        llm = VGroup(box, box_label)
        feed = Arrow(sentence.get_bottom() + DOWN * 0.1, box.get_top() + UP * 0.05,
                     buff=0.1, color=style.DIM, stroke_width=3,
                     max_tip_length_to_length_ratio=0.25)

        # ---- the training loop around the box ----
        a, b = 3.6, 1.75  # ellipse half-axes
        names = ["Predict", "Loss", "Backprop", "Update"]
        colors = [style.BLUE, style.RED, style.RED, style.GREEN]
        angles = [PI / 2, 0, -PI / 2, -PI]  # clockwise from the top
        pt = lambda th: center + np.array([a * np.cos(th), b * np.sin(th), 0])
        labels = VGroup(*[
            T(n, size=style.BODY_SIZE, color=c).move_to(pt(th))
            for n, c, th in zip(names, colors, angles)
        ])
        arrows = VGroup()
        for i in range(4):
            t0, t1 = angles[i] - 0.42, angles[i] - PI / 2 + 0.42
            if i % 2 == 1:  # leaving a side label needs more room (wider text)
                t0 = angles[i] - 0.55
            else:
                t1 = angles[i] - PI / 2 + 0.55
            arc = ParametricFunction(lambda u, t0=t0, t1=t1: pt(t0 + (t1 - t0) * u),
                                     t_range=[0, 1], color=style.DIM, stroke_width=3)
            tip_dir = pt(t1) - pt(t1 + 0.05)
            tip = ArrowTriangleFilledTip(color=style.DIM, length=0.2, width=0.2)
            tip.rotate(np.arctan2(tip_dir[1], tip_dir[0]))
            tip.move_to(pt(t1) + 0.08 * tip_dir / np.linalg.norm(tip_dir))
            arrows.add(VGroup(arc, tip))

        # step counter (Text, not Integer: Integer needs LaTeX)
        step_val = ValueTracker(0)
        def make_counter():
            g = VGroup(T("step", size=style.LABEL_SIZE + 4, color=style.DIM),
                       T(f"{int(round(step_val.get_value())):>3d}", size=style.BODY_SIZE + 4,
                         color=style.YELLOW)).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
            return g.to_corner(DR, buff=0.6)
        counter_grp = always_redraw(make_counter)

        # ---- beat 1: sentence types in, box appears ----
        with self.voiceover(text="A language model learns by guessing the next word.") as tr:
            self.play(AddTextLetterByLetter(prompt), run_time=1.2)
            self.play(FadeIn(blank), GrowArrow(feed), FadeIn(llm, shift=UP * 0.2), run_time=0.8)
            self.play(Indicate(box, color=style.YELLOW, scale_factor=1.08), run_time=0.7)
            self.wait(max(0.0, tr.duration - 2.7))

        # ---- beat 2: a wrong guess ----
        wrong = T("eat", size=style.HEADING_SIZE, color=style.RED).move_to(blank, aligned_edge=DL)
        cross = cross_mark().next_to(wrong, RIGHT, buff=0.35)
        with self.voiceover(text="At first, its guesses are wrong.") as tr:
            self.play(ReplacementTransform(blank, wrong), run_time=0.6)
            self.play(FadeIn(cross, scale=1.4), run_time=0.5)
            self.wait(max(0.0, tr.duration - 1.1))
        self.wait(0.4)

        # ---- beat 3: the loop is drawn one stage at a time ----
        with self.voiceover(text="So it measures how wrong it was, and nudges its weights.") as tr:
            self.play(FadeOut(feed), run_time=0.4)  # make room for the loop
            per = max(0.9, (tr.duration - 0.6) / 4)
            for lab, ar in zip(labels, arrows):
                self.play(FadeIn(lab, scale=0.8), run_time=per * 0.45)
                self.play(Create(ar), run_time=per * 0.55)
        self.wait(0.6)

        # ---- beat 4: the loop spins 3 times while the counter runs to 300 ----
        theta = ValueTracker(PI / 2)
        dot = Dot(pt(PI / 2), radius=0.11, color=style.YELLOW)
        dot.add_updater(lambda d: d.move_to(pt(theta.get_value())))
        glow = Dot(radius=0.18, color=style.YELLOW, fill_opacity=0.3)  # trailing halo
        glow.add_updater(lambda d: d.move_to(pt(theta.get_value() + 0.12)))
        step_val.add_updater(lambda v: v.set_value(300 * (PI / 2 - theta.get_value()) / (3 * TAU)))

        def light(m):  # highlight the stage the dot is passing
            th = theta.get_value()
            for lab, ang in zip(labels, angles):
                d = abs((th - ang + PI) % TAU - PI)
                lab.set_opacity(1.0 if d < 0.6 else 0.45)

        with self.voiceover(text="And it repeats this loop, thousands of times.") as tr:
            self.play(FadeIn(counter_grp), FadeIn(dot), FadeIn(glow), run_time=0.4)
            self.add(step_val)
            labels.add_updater(light)
            spin = max(4.0, tr.duration - 0.4)
            self.play(theta.animate.set_value(PI / 2 - 3 * TAU), run_time=spin, rate_func=smooth)
            for m in (dot, glow, step_val, labels):
                m.clear_updaters()
            step_val.set_value(300)
            labels.set_opacity(1)

        # ---- beat 5: the right guess ----
        right = T("be", size=style.HEADING_SIZE, color=style.GREEN).move_to(wrong, aligned_edge=DL)
        check = check_mark().next_to(right, RIGHT, buff=0.35)
        with self.voiceover(text="Until it guesses right.") as tr:
            self.play(FadeOut(dot), FadeOut(glow),
                      ReplacementTransform(wrong, right), ReplacementTransform(cross, check),
                      Flash(box, color=style.GREEN, line_length=0.3), run_time=0.9)
            self.wait(max(0.0, tr.duration - 0.9))
        self.wait(0.8)

        # ---- beat 6: the title ----
        title = T("How a Llama Learns", size=style.TITLE_SIZE, color=style.BLUE)
        with self.voiceover(text="Let's see how.") as tr:
            loop = VGroup(labels, arrows, llm, counter_grp)
            answer = VGroup(prompt, right, check)
            target = answer.copy().scale(0.75, scale_stroke=True).move_to(DOWN * 1.3)
            target[0].set_opacity(0.6)
            target[1].set_opacity(0.6)
            target[2].set_stroke(opacity=0.6)  # the check is a stroke-only shape
            self.play(FadeOut(loop, scale=0.9), Transform(answer, target), run_time=0.7)
            self.play(Write(title), run_time=max(1.0, tr.duration - 0.7))
        self.wait(1.2)
        self.fade_out_all()
