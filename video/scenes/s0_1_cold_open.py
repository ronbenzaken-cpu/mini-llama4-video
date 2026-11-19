"""Scene 0.1 - Opening: what this video is about.

Built together with the user, step by step; it becomes the model scene for all subagents.
Step 1 (this version): title -> what a language model does -> roadmap of the video.

Render:  .venv/Scripts/python -m manim -ql video/scenes/s0_1_cold_open.py Scene_0_1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, FadeIn, FadeOut, GrowArrow,  # noqa: E402
                   ReplacementTransform, VGroup, Write)

from video.common import style  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402
from video.common.tokens import token_box, token_row  # noqa: E402


class Scene_0_1(NarratedScene):
    def roadmap_item(self, number: str, title: str, subtitle: str) -> VGroup:
        num = self.text(number, size=style.HEADING_SIZE, color=style.YELLOW)
        head = self.text(title, size=style.BODY_SIZE, color=style.TEXT)
        sub = self.text(subtitle, size=style.LABEL_SIZE, color=style.DIM)
        words = VGroup(head, sub).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        return VGroup(num, words).arrange(RIGHT, buff=0.4)

    def construct(self):
        # ---------- 1. title: what is this video? ----------
        title = self.text("How a Llama Learns", size=style.TITLE_SIZE, color=style.BLUE)
        with self.voiceover(text="Today we'll learn how a language model learns "
                                 "to guess the next word correctly.") as t:
            self.play(Write(title), run_time=t.duration * 0.7)
        self.read_pause()

        # ---------- 2. what does a language model do? ----------
        heading = self.heading("What does a language model do?")
        prompt = token_row(["To", "be", "or", "not", "to"])
        blank = token_box("?", color=style.DIM)
        sentence = VGroup(prompt, blank).arrange(RIGHT, buff=0.2).move_to(UP * 0.9)
        model = token_box("language model", color=style.BLUE).move_to(DOWN * 1.4)
        arrow_in = Arrow(prompt.get_bottom(), model.get_top() + LEFT * 0.6, buff=0.15, color=style.DIM)
        arrow_out = Arrow(model.get_top() + RIGHT * 0.6, blank.get_bottom(), buff=0.15, color=style.YELLOW)
        answer = token_box("be", color=style.GREEN).move_to(blank)

        with self.voiceover(text="You give it the start of a sentence,") as t:
            self.play(ReplacementTransform(title, heading), run_time=t.duration * 0.3)
            self.play(*[FadeIn(b, shift=UP * 0.2) for b in prompt], FadeIn(blank),
                      lag_ratio=0.15, run_time=t.duration * 0.6)
        with self.voiceover(text="and it predicts the word that comes next.") as t:
            self.play(FadeIn(model), GrowArrow(arrow_in), run_time=t.duration * 0.4)
            self.play(GrowArrow(arrow_out), run_time=t.duration * 0.25)
            self.play(ReplacementTransform(blank, answer), run_time=t.duration * 0.3)
        self.read_pause(3)  # let the viewer see the full example once

        # ---------- 3. roadmap: what will we do in this video? ----------
        self.play(*[FadeOut(m) for m in (prompt, answer, model, arrow_in, arrow_out)], run_time=0.4)
        road_heading = self.heading("What we'll do")
        items = VGroup(
            self.roadmap_item("1", "Llama 4", "Meta's open model: how it is built"),
            self.roadmap_item("2", "Mini Llama", "the same design, small enough for a laptop"),
            self.roadmap_item("3", "The training code", "line by line: how it learns"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to(style.CONTENT_CENTER)

        with self.voiceover(text="We'll use Meta's open model, Llama. First, we'll see how it is built.") as t:
            self.play(ReplacementTransform(heading, road_heading), run_time=t.duration * 0.25)
            self.play(FadeIn(items[0], shift=RIGHT * 0.3), run_time=t.duration * 0.4)
        self.read_pause()
        with self.voiceover(text="Then, our own mini version, small enough to train on a laptop.") as t:
            self.play(FadeIn(items[1], shift=RIGHT * 0.3), run_time=t.duration * 0.4)
        self.read_pause()
        with self.voiceover(text="And finally, the training code, line by line, to see exactly how it learns.") as t:
            self.play(FadeIn(items[2], shift=RIGHT * 0.3), run_time=t.duration * 0.4)
        self.read_pause(4)

        self.fade_out_all()
