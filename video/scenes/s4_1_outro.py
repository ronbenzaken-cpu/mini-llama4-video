"""Scene 4.1 - Outro: before vs. after training, then the loop and the title.

Echoes scene 0.1: it ends on the same title card ("How a Llama Learns" + subtitle).
All text samples and probabilities come from TRACE (real untrained vs. trained model).

Preview:  .venv/Scripts/python -m manim -ql video/scenes/s4_1_outro.py Scene_4_1
Final:    .venv/Scripts/python -m manim -qh video/scenes/s4_1_outro.py Scene_4_1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manim import (DOWN, LEFT, RIGHT, UP, Create, CurvedArrow, FadeIn, FadeOut, LaggedStart,  # noqa: E402
                   RoundedRectangle, Transform, ValueTracker, VGroup, Write, always_redraw,
                   linear, smooth)

from video.common import style  # noqa: E402
from video.common.data import TRACE  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402
from video.common.tokens import token_box  # noqa: E402

SCROLL = UP * 8           # one "page" up: used for section changes (same as 0.1)
MAX_CHARS = 41            # excerpt width: the longest trained line we show
N_STEPS = int(TRACE["loss_curve"][-1][0]) + 1   # 300 training steps
P_BEFORE = TRACE["untrained"]["p_target"] * 100  # 1.2 %
P_AFTER = TRACE["trained"]["p_target"] * 100     # 74 %


def excerpt(sample: str, n_lines: int = 3) -> list[str]:
    """First non-empty lines of a real sample, cut to MAX_CHARS so both cards are equally wide."""
    lines = [ln for ln in sample.split("\n") if ln.strip()]
    return [ln[:MAX_CHARS] for ln in lines[:n_lines]]


class Scene_4_1(NarratedScene):
    def sample_card(self, label: str, color, lines: list[str], text_color, width: float) -> VGroup:
        """A label column ("step 0") + a rounded card holding the sample lines."""
        rows = VGroup(*[self.text(ln, size=28, color=text_color) for ln in lines])
        # baseline-safe rows: fixed line pitch, left aligned
        for i, r in enumerate(rows):
            r.move_to(DOWN * 0.55 * i, aligned_edge=LEFT)
            r.align_to(rows[0], LEFT)
        card = RoundedRectangle(corner_radius=0.15, width=width, height=rows.height + 0.6,
                                stroke_color=color, fill_color=style.GREY_BOX, fill_opacity=1)
        rows.move_to(card).align_to(card, LEFT).shift(RIGHT * 0.35)
        name = self.text(label, size=style.BODY_SIZE, color=color)
        name.next_to(card, LEFT, buff=0.4).align_to(card, UP).shift(DOWN * 0.2)
        return VGroup(name, card, rows)

    def construct(self):
        # ================= 1. Before vs. after =================
        heading = self.heading("Before vs. after training")
        self.play(Write(heading), run_time=style.LEAD_IN)

        lines_b, lines_a = excerpt(TRACE["samples"]["untrained"]), excerpt(TRACE["samples"]["trained"])
        card_w = max(self.text(ln, size=28).width for ln in lines_b + lines_a) + 0.7
        before = self.sample_card("step 0", style.RED, lines_b, style.DIM, card_w)
        after = self.sample_card(f"step {N_STEPS}", style.GREEN, lines_a, style.TEXT, card_w)
        cards = VGroup(before, after).arrange(DOWN, buff=0.75, aligned_edge=RIGHT)
        cards.move_to(style.CONTENT_CENTER + UP * 0.2)

        p_before = self.text(f"p(right next char) {P_BEFORE:.1f}%", size=style.LABEL_SIZE, color=style.RED)
        p_after = self.text(f"p(right next char) {P_AFTER:.0f}%", size=style.LABEL_SIZE, color=style.GREEN)
        p_before.next_to(before[1], DOWN, buff=0.12).align_to(before[1], RIGHT)
        p_after.next_to(after[1], DOWN, buff=0.12).align_to(after[1], RIGHT)

        with self.voiceover(text='So, what did <bookmark mark="steps"/>three hundred steps of training change? '
                                 'We ask the model to <bookmark mark="write"/>write a play, '
                                 'starting from the name Romeo.'):
            self.wait_until_bookmark("write")
            self.play(FadeIn(before[0]), FadeIn(before[1]), FadeIn(after[0]), FadeIn(after[1]),
                      run_time=0.6)
        self.read_pause()

        def type_in(rows):
            chars = [c for r in rows for c in r]
            return LaggedStart(*[FadeIn(c) for c in chars], lag_ratio=0.03)

        with self.voiceover(text='<bookmark mark="b"/>Before training, it writes random letters.'):
            self.wait_until_bookmark("b")
            self.play(type_in(before[2]), run_time=2.0, rate_func=linear)
        self.read_pause()
        with self.voiceover(text='<bookmark mark="a"/>After training, it writes words, lines and speakers. '
                                 'Not quite Shakespeare, but close.'):
            self.wait_until_bookmark("a")
            self.play(type_in(after[2]), run_time=2.0, rate_func=linear)
        self.read_pause(2)

        with self.voiceover(text='And after "to be or", the chance it gives the right next character '
                                 'grew from <bookmark mark="p0"/>one point two percent '
                                 'to <bookmark mark="p1"/>seventy-four percent.'):
            self.wait_until_bookmark("p0")
            self.play(FadeIn(p_before, shift=UP * 0.15), run_time=0.5)
            self.wait_until_bookmark("p1")
            self.play(FadeIn(p_after, shift=UP * 0.15), run_time=0.5)
        self.read_pause(3)

        # ================= 2. The loop =================
        page1 = VGroup(heading, cards, p_before, p_after)
        loop_heading = self.heading("The training loop").shift(-SCROLL)

        center = style.CONTENT_CENTER + DOWN * 0.2
        spots = [center + UP * 1.9, center + RIGHT * 3.6, center + DOWN * 1.9, center + LEFT * 3.6]
        words = ["Predict", "Measure", "Blame", "Nudge"]
        colors = [style.BLUE, style.RED, style.RED, style.GREEN]
        notes = ["lines 1-6", "line 7: loss", "line 8: backward", "line 9: step"]
        nodes = VGroup(*[token_box(w, color=c).move_to(p) for w, c, p in zip(words, colors, spots)])
        subs = VGroup(*[self.text(n, size=style.LABEL_SIZE, color=style.DIM).next_to(nd, DOWN, buff=0.12)
                        for n, nd in zip(notes, nodes)])
        # clockwise arrows between neighbours, bending outward
        # arrows leave / enter the side nodes below their sublabels, so no arrow crosses a label
        ends = [(nodes[0].get_right(), nodes[1].get_top()), (subs[1].get_bottom(), nodes[2].get_right()),
                (nodes[2].get_left(), subs[3].get_bottom()), (nodes[3].get_top(), nodes[0].get_left())]
        arrows = VGroup(*[CurvedArrow(a + d, b, angle=-1.2, color=style.DIM, tip_length=0.2)
                          for (a, b), d in zip(ends, [RIGHT * 0.1, DOWN * 0.08, LEFT * 0.1, UP * 0.1])])

        steps = ValueTracker(0)
        counter = always_redraw(lambda: self.text(f"step {int(steps.get_value())}",
                                                  size=style.BODY_SIZE, color=style.YELLOW)
                                .move_to(center))

        with self.voiceover(text='Every step of that came from <bookmark mark="one"/>one simple loop.'):
            self.play(page1.animate.shift(SCROLL), loop_heading.animate.shift(SCROLL),
                      run_time=0.9, rate_func=smooth)
            self.remove(page1)
            self.wait_until_bookmark("one")
            self.play(*[Create(a) for a in arrows], run_time=0.8)
        self.read_pause()

        def light(i):
            return [FadeIn(nodes[i], scale=1.15), FadeIn(subs[i]), arrows[i].animate.set_color(colors[i])]

        with self.voiceover(text='<bookmark mark="w0"/>Predict, <bookmark mark="w1"/>measure, '
                                 '<bookmark mark="w2"/>blame, <bookmark mark="w3"/>nudge.'):
            for i in range(4):
                self.wait_until_bookmark(f"w{i}")
                self.play(*light(i), run_time=0.35)
        self.read_pause()

        with self.voiceover(text='That loop, <bookmark mark="rep"/>repeated, '
                                 'is <bookmark mark="how"/>how a Llama learns.'):
            self.wait_until_bookmark("rep")
            self.add(counter)
            self.play(steps.animate.set_value(N_STEPS), run_time=1.6, rate_func=linear)
            counter.clear_updaters()
            # ================= 3. The title (same card as scene 0.1) =================
            loop = VGroup(nodes, arrows)
            title = self.text("How a Llama Learns", size=style.TITLE_SIZE, color=style.BLUE)
            subtitle = self.text("guessing the next word, correctly", size=style.BODY_SIZE, color=style.DIM)
            VGroup(title, subtitle).arrange(DOWN, buff=0.35).move_to(UP * 2.2)
            self.wait_until_bookmark("how")
            self.play(Transform(loop_heading, title),
                      loop.animate.scale(0.7).move_to(DOWN * 1.5), FadeOut(subs), FadeOut(counter),
                      run_time=1.0)
            self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.6)
        self.read_pause(4)

        self.fade_out_all()
