"""5-second stack test: one title, one narrated sentence.

    manim -ql video/test_clip.py TestClip
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manim import DOWN, FadeIn, Text, Write  # noqa: E402

from video.common import style  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402


class TestClip(NarratedScene):
    def construct(self):
        title = Text("How a Llama Learns", font=style.FONT, font_size=style.TITLE_SIZE, color=style.BLUE)
        sub = Text("stack test", font=style.FONT, font_size=style.BODY_SIZE, color=style.DIM)
        sub.next_to(title, DOWN)
        with self.voiceover(text="This is a test of the animation and narration stack.") as tracker:
            self.play(Write(title), run_time=tracker.duration * 0.6)
            self.play(FadeIn(sub))
        self.wait(1)
        self.fade_out_all()
