"""Scene 0.1 - Opening: what this video is about.

Built together with the user, step by step; it becomes the model scene for all subagents.
Step 1: title -> what a language model does -> roadmap of the video.

Techniques shown here (copy them):
  * bookmarks: animations start on the exact spoken word   (self.wait_until_bookmark)
  * morphs instead of fade-out/fade-in                       (FadeTransform, .animate)
  * section change = scroll: old content slides up, new content slides in from below

Preview:  .venv/Scripts/python -m manim -ql video/scenes/s0_1_cold_open.py Scene_0_1
Final:    .venv/Scripts/python -m manim -qh video/scenes/s0_1_cold_open.py Scene_0_1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, FadeIn, FadeTransform, GrowArrow,  # noqa: E402
                   LaggedStart, SurroundingRectangle, VGroup, Write, Create, smooth)

from video.common import style  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402
from video.common.tokens import token_box, token_row  # noqa: E402

SCROLL = UP * 8  # one "page" up: used for section changes


class Scene_0_1(NarratedScene):
    def roadmap_item(self, number: str, title: str, subtitle: str) -> VGroup:
        num = self.text(number, size=style.HEADING_SIZE, color=style.YELLOW)
        head = self.text(title, size=style.BODY_SIZE, color=style.TEXT)
        sub = self.text(subtitle, size=style.LABEL_SIZE, color=style.DIM)
        words = VGroup(head, sub).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        return VGroup(num, words).arrange(RIGHT, buff=0.4)

    def construct(self):
        # ================= 1. Title: what is this video? =================
        title = self.text("How a Llama Learns", size=style.TITLE_SIZE, color=style.BLUE)
        subtitle = self.text("guessing the next word, correctly", size=style.BODY_SIZE, color=style.DIM)
        subtitle.next_to(title, DOWN, buff=0.35)

        with self.voiceover(text='Today we\'ll learn how a <bookmark mark="lm"/>language model learns '
                                 'to <bookmark mark="guess"/>guess the next word correctly.'):
            self.play(Write(title), run_time=1.2)
            self.wait_until_bookmark("guess")
            self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.8)
        self.read_pause()

        # ================= 2. What does a language model do? =================
        heading = self.heading("What does a language model do?").shift(-SCROLL)
        prompt = token_row(["To", "be", "or", "not", "to"])
        blank = token_box("?", color=style.DIM)
        VGroup(prompt, blank).arrange(RIGHT, buff=0.2).move_to(UP * 0.9)
        model = token_box("language model", color=style.BLUE).move_to(DOWN * 1.5)
        arrow_in = Arrow(prompt.get_bottom(), model.get_top() + LEFT * 0.7, buff=0.15, color=style.DIM)
        arrow_out = Arrow(model.get_top() + RIGHT * 0.7, blank.get_bottom(), buff=0.15, color=style.YELLOW)
        answer = token_box("be", color=style.GREEN).move_to(blank)

        with self.voiceover(text='You give it <bookmark mark="start"/>the start of a sentence,'):
            # section change = scroll up: the title leaves, the next heading arrives from below
            self.play(VGroup(title, subtitle).animate.shift(SCROLL), heading.animate.shift(SCROLL),
                      run_time=0.9, rate_func=smooth)
            self.remove(title, subtitle)
            self.wait_until_bookmark("start")
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.25) for b in prompt], lag_ratio=0.18),
                      FadeIn(blank), run_time=1.0)
        with self.voiceover(text='and it <bookmark mark="pred"/>predicts the word '
                                 'that <bookmark mark="next"/>comes next.'):
            self.play(GrowArrow(arrow_in), FadeIn(model, shift=UP * 0.2), run_time=0.6)
            self.wait_until_bookmark("pred")
            self.play(model[0].animate.set_stroke(style.YELLOW), GrowArrow(arrow_out), run_time=0.5)
            self.wait_until_bookmark("next")
            self.play(FadeTransform(blank, answer), model[0].animate.set_stroke(style.BLUE), run_time=0.5)
        self.read_pause(4)  # let the viewer see the whole example once

        # ================= 3. Roadmap: what will we do? =================
        example = VGroup(heading, prompt, answer, model, arrow_in, arrow_out)
        road_heading = self.heading("What we'll do").shift(-SCROLL)
        items = VGroup(
            self.roadmap_item("1", "Llama 4", "Meta's open model: how it is built"),
            self.roadmap_item("2", "Mini Llama", "the same design, small enough for a laptop"),
            self.roadmap_item("3", "The training code", "line by line: how it learns"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to(style.CONTENT_CENTER)
        for it in items:
            it.set_opacity(0)

        with self.voiceover(text='We\'ll use <bookmark mark="meta"/>Meta\'s open model, Llama. '
                                 'First, we\'ll see <bookmark mark="built"/>how it is built.'):
            # section change = scroll up: the example leaves, the roadmap heading arrives
            self.play(example.animate.shift(SCROLL), road_heading.animate.shift(SCROLL),
                      run_time=0.9, rate_func=smooth)
            self.remove(example)
            self.wait_until_bookmark("meta")
            self.play(items[0].animate.set_opacity(1), run_time=0.6)
        self.read_pause()
        with self.voiceover(text='Then, our own <bookmark mark="mini"/>mini version, '
                                 'small enough to train on a laptop.'):
            self.wait_until_bookmark("mini")
            self.play(items[1].animate.set_opacity(1), run_time=0.6)
        self.read_pause()
        with self.voiceover(text='And finally, <bookmark mark="code"/>the training code, line by line, '
                                 'to see exactly how it learns.'):
            self.wait_until_bookmark("code")
            self.play(items[2].animate.set_opacity(1), run_time=0.6)
        self.read_pause()

        # hand-off cue: highlight where the video goes next
        focus = SurroundingRectangle(items[0], color=style.YELLOW, buff=0.2, corner_radius=0.1)
        with self.voiceover(text='Let\'s <bookmark mark="go"/>start with Llama.'):
            self.wait_until_bookmark("go")
            self.play(Create(focus), items[1:].animate.set_opacity(0.35), run_time=0.7)
        self.read_pause(3)

        self.fade_out_all()
