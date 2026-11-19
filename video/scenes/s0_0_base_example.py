"""BASE EXAMPLE SCENE: the reference every scene agent copies from. Not part of the final video.

It shows every convention in SCRIPT.md, in order:
  1. file/import template           5. one idea per narration block + read pause
  2. heading at the standard spot   6. colors used by meaning (yellow = focus, red = wrong, green = right)
  3. objects built from code only   7. live counter without LaTeX (ValueTracker + always_redraw)
  4. sync to voice with tracker     8. standard ending: fade_out_all()

Render:  .venv/Scripts/python -m manim -ql video/scenes/s0_0_base_example.py Scene_0_0
"""

# ---- 1. template: make the repo root importable, then import the shared style + base class ----
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, FadeIn, FadeOut, Indicate,  # noqa: E402
                   ReplacementTransform, SurroundingRectangle, ValueTracker,
                   VGroup, Write, always_redraw, linear)

from video.common import style  # noqa: E402
from video.common.marks import check_mark, cross_mark  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402
from video.common.tokens import token_box  # noqa: E402


class Scene_0_0(NarratedScene):
    # Reuse shared builders (video/common/*) instead of redrawing common things.
    # Scene-specific builders go in small methods here; construct() only animates.
    token_box = staticmethod(token_box)

    def construct(self):
        # ---- 2. heading: every scene opens with a short heading at style.HEADING_POS ----
        with self.voiceover(text="Every scene starts on an empty screen, with a short heading.") as t:
            heading = self.heading("The next-word game")
            self.play(Write(heading), run_time=t.duration * 0.6)
        self.read_pause()

        # ---- 3 + 4. build from code, and stretch animations over the voice with t.duration ----
        words = ["To", "be", "or", "not", "to"]
        row = VGroup(*[self.token_box(w) for w in words]).arrange(RIGHT, buff=0.2)
        blank = self.token_box("?", color=style.DIM)
        sentence = VGroup(row, blank).arrange(RIGHT, buff=0.2).move_to(style.CONTENT_CENTER + UP * 0.8)

        with self.voiceover(text="Then we add one idea at a time. Here, a sentence with a missing next word.") as t:
            self.play(*[FadeIn(b, shift=UP * 0.2) for b in row], lag_ratio=0.15, run_time=t.duration * 0.6)
            self.play(FadeIn(blank), run_time=t.duration * 0.2)
        self.read_pause()

        # ---- 6. colors by meaning: yellow = look here ----
        focus = SurroundingRectangle(blank, color=style.YELLOW, buff=0.08)
        with self.voiceover(text="Yellow always means: look here.") as t:
            self.play(Create(focus), run_time=t.duration * 0.5)
            self.play(Indicate(blank, color=style.YELLOW), run_time=t.duration * 0.4)
        self.read_pause()

        # ---- red = wrong, green = right; marks are drawn shapes (Consolas has no check/cross glyph) ----
        model = self.token_box("model", color=style.BLUE).next_to(sentence, DOWN, buff=1.3)
        arrow = Arrow(model.get_top(), blank.get_bottom(), buff=0.1, color=style.DIM)
        wrong = self.token_box("eat", color=style.RED).move_to(blank)
        x = cross_mark().next_to(wrong, RIGHT, buff=0.2)
        with self.voiceover(text="Red means wrong. The model's first guess is eat.") as t:
            self.play(FadeIn(model), Create(arrow), run_time=t.duration * 0.4)
            self.play(ReplacementTransform(blank, wrong), FadeIn(x), run_time=t.duration * 0.4)
        self.read_pause()

        right = self.token_box("be", color=style.GREEN).move_to(wrong)
        v = check_mark().next_to(right, RIGHT, buff=0.2)
        with self.voiceover(text="Green means right. After training, it guesses be.") as t:
            self.play(ReplacementTransform(wrong, right), ReplacementTransform(x, v), run_time=t.duration * 0.5)
        self.read_pause()

        # ---- 7. numbers without LaTeX: a Text redrawn from a ValueTracker ----
        steps = ValueTracker(0)
        counter = always_redraw(lambda: self.text(f"training steps: {int(steps.get_value())}",
                                                  size=style.LABEL_SIZE, color=style.DIM)
                                .to_corner(DOWN + RIGHT, buff=0.5))
        with self.voiceover(text="Numbers are plain text driven by a tracker, because there is no LaTeX here.") as t:
            self.add(counter)
            self.play(steps.animate.set_value(300), run_time=t.duration * 0.8, rate_func=linear)
        self.read_pause(2)

        # ---- 8. every scene ends the same way ----
        with self.voiceover(text="And every scene ends by fading out.") as t:
            self.wait(t.duration)
        self.fade_out_all()
