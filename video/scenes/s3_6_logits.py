"""Scene 3.6 - Line 6: logits. The last vector becomes 65 scores, then 65 probabilities.

Real data (video.common.data.TRACE["untrained"]):
  * last_token_probs: the untrained model's 65 probabilities for the character after "To be or"
  * target_id: the correct next character (a space)
The logit bars are drawn as log(p) minus its mean: softmax ignores a constant shift, so these are
exactly the model's logits up to that shift, and softmax of them gives back the real probabilities.

Preview:  .venv/Scripts/python -m manim -ql video/scenes/s3_6_logits.py Scene_3_6
Final:    .venv/Scripts/python -m manim -qh video/scenes/s3_6_logits.py Scene_3_6
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, DashedLine, FadeIn,  # noqa: E402
                   GrowArrow, GrowFromPoint, LaggedStart, Line, Rectangle, RoundedRectangle,
                   SurroundingRectangle, Transform, VGroup, VMobject, Write)

from video.common import style  # noqa: E402
from video.common.code_panel import CodePanel  # noqa: E402
from video.common.data import TRACE, show_char  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402
from video.common.tokens import word  # noqa: E402

UNTRAINED = TRACE["untrained"]
PROBS = UNTRAINED["last_token_probs"]               # 65 real probabilities
TARGET = UNTRAINED["target_id"]                     # id of the right next character
VOCAB = TRACE["vocab"]
CHARS = TRACE["tokens"]["chars"]                    # 'T','o',' ','b','e',' ','o','r'
N = len(PROBS)                                      # 65

LOGP = [math.log(p) for p in PROBS]
MEAN_LOGP = sum(LOGP) / N
LOGITS = [lp - MEAN_LOGP for lp in LOGP]            # logits up to a constant shift

# ---- chart geometry (right-hand animation area) ----
CHART_LEFT = -0.4
PITCH = 0.085                                       # 65 bars -> 5.5 units wide
BAR_W = 0.06
LOGIT_BASE = -1.55                                  # zero line of the logit chart
LOGIT_SCALE = 1.3                                   # units per logit
PROB_BASE = -2.55                                   # zero line of the probability chart
PROB_SCALE = 68.0                                   # units per probability (max p 0.026 -> 1.8)


def bar_x(i: int) -> float:
    return CHART_LEFT + PITCH * (i + 0.5)


def logit_bar(i: int) -> Rectangle:
    h = max(abs(LOGITS[i]) * LOGIT_SCALE, 0.01)
    r = Rectangle(width=BAR_W, height=h, stroke_width=0, fill_color=style.BLUE, fill_opacity=0.9)
    r.move_to([bar_x(i), LOGIT_BASE + math.copysign(h / 2, LOGITS[i]), 0])
    return r


def prob_bar(i: int) -> Rectangle:
    h = PROBS[i] * PROB_SCALE
    r = Rectangle(width=BAR_W, height=h, stroke_width=0, fill_color=style.BLUE, fill_opacity=0.9)
    r.move_to([bar_x(i), PROB_BASE + h / 2, 0])
    return r


class Scene_3_6(NarratedScene):
    def char_label(self, ch: str, size=style.LABEL_SIZE, color=style.TEXT):
        """A character as a label; the space is drawn as an open bracket (Consolas lacks the glyph)."""
        if ch != " ":
            return word(show_char(ch), size=size, color=color)
        w, h = 0.11 * size / style.LABEL_SIZE, 0.09 * size / style.LABEL_SIZE
        m = VMobject(stroke_color=color, stroke_width=3)
        m.set_points_as_corners([LEFT * w + UP * h, LEFT * w, RIGHT * w, RIGHT * w + UP * h])
        return m

    def small_token(self, ch: str, color=style.DIM) -> VGroup:
        label = self.char_label(ch)
        box = RoundedRectangle(corner_radius=0.08, width=0.5, height=0.55, stroke_color=color,
                               fill_color=style.GREY_BOX, fill_opacity=1)
        label.move_to(box)
        if ch == " ":
            label.shift(DOWN * 0.08)
        return VGroup(box, label)

    def vector(self) -> VGroup:
        """The last character's vector, drawn horizontally (shading is decorative)."""
        shades = [0.35, 0.8, 0.55, 0.95, 0.25, 0.7, 0.45, 0.85, 0.3, 0.6, 0.9, 0.4]
        cells = VGroup(*[Rectangle(width=0.22, height=0.3, stroke_color=style.BG, stroke_width=1.5,
                                   fill_color=style.BLUE, fill_opacity=s) for s in shades])
        cells.arrange(RIGHT, buff=0)
        return cells

    def construct(self):
        anim_x = CHART_LEFT + PITCH * N / 2                 # horizontal center of the right area

        # ================= 1. Heading + code panel (lead-in) =================
        heading = self.heading("Line 6: logits")
        panel = CodePanel().focus_now(5)
        self.play(Write(heading), FadeIn(panel), run_time=style.LEAD_IN)

        with self.voiceover(text='Now the vectors leave the blocks. <bookmark mark="line"/>Line six turns '
                                 'them into a guess for the next character.'):
            self.wait_until_bookmark("line")
            self.play(panel.focus(6), run_time=0.6)
        self.read_pause()

        # ================= 2. The last character's vector =================
        tokens = VGroup(*[self.small_token(c) for c in CHARS]).arrange(RIGHT, buff=0.1)
        tokens.move_to([anim_x, 2.35, 0])
        last = tokens[-1]
        vec = self.vector().move_to([anim_x, 1.35, 0])
        vec_label = self.text("128 numbers", size=style.LABEL_SIZE, color=style.DIM)
        vec_label.next_to(vec, RIGHT, buff=0.25)
        drop = Arrow(last.get_bottom(), vec.get_top() + RIGHT * 1.0, buff=0.08, color=style.DIM,
                     stroke_width=3, max_tip_length_to_length_ratio=0.25)

        with self.voiceover(text='Take the vector of the <bookmark mark="last"/>last character, r.'):
            self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.15) for t in tokens], lag_ratio=0.1),
                      run_time=0.8)
            self.wait_until_bookmark("last")
            self.play(last[0].animate.set_stroke(style.YELLOW), run_time=0.4)
            self.play(GrowArrow(drop), FadeIn(vec, shift=DOWN * 0.2), FadeIn(vec_label), run_time=0.6)
        self.read_pause()

        # ================= 3. Output matrix -> 65 scores (logits) =================
        matrix = RoundedRectangle(corner_radius=0.1, width=4.6, height=0.6, stroke_color=style.BLUE,
                                  fill_color=style.GREY_BOX, fill_opacity=1)
        matrix.move_to([anim_x, 0.35, 0])
        matrix_label = self.text("output matrix  128 → 65", size=style.LABEL_SIZE)
        matrix_label.move_to(matrix)
        matrix_grp = VGroup(matrix, matrix_label)
        into = Arrow(vec.get_bottom(), matrix.get_top(), buff=0.06, color=style.DIM, stroke_width=3,
                     max_tip_length_to_length_ratio=0.3)

        zero = Line([CHART_LEFT, LOGIT_BASE, 0], [CHART_LEFT + PITCH * N, LOGIT_BASE, 0],
                    color=style.DIM, stroke_width=1.5)
        logit_bars = VGroup(*[logit_bar(i) for i in range(N)])
        caption = self.text("65 scores (logits)", size=style.LABEL_SIZE, color=style.BLUE)
        caption.move_to([anim_x, -3.3, 0])

        with self.voiceover(text='It is normalized, then <bookmark mark="mat"/>multiplied by the output '
                                 'matrix. Out come <bookmark mark="scores"/>65 scores, one for each '
                                 'possible next character.'):
            self.wait_until_bookmark("mat")
            self.play(GrowArrow(into), FadeIn(matrix_grp, shift=DOWN * 0.15), run_time=0.6)
            self.play(vec.copy().animate.move_to(matrix).set_opacity(0), run_time=0.5)
            self.wait_until_bookmark("scores")
            self.play(Create(zero), run_time=0.3)
            self.play(LaggedStart(*[GrowFromPoint(b, [b.get_x(), LOGIT_BASE, 0]) for b in logit_bars],
                                  lag_ratio=0.03),
                      FadeIn(caption, shift=UP * 0.15), run_time=1.4)
        self.read_pause(2)

        # ================= 4. Softmax -> probabilities =================
        prob_bars = VGroup(*[prob_bar(i) for i in range(N)])
        prob_axis = Line([CHART_LEFT, PROB_BASE, 0], [CHART_LEFT + PITCH * N, PROB_BASE, 0],
                         color=style.DIM, stroke_width=1.5)
        caption2 = self.text("65 probabilities, adding up to 1", size=style.LABEL_SIZE, color=style.BLUE)
        caption2.move_to(caption)

        with self.voiceover(text='<bookmark mark="soft"/>Softmax turns these scores into probabilities '
                                 'that add up to one.'):
            self.wait_until_bookmark("soft")
            self.play(Transform(logit_bars, prob_bars), Transform(zero, prob_axis),
                      caption.animate.shift(UP * 0.35).set_opacity(0),
                      FadeIn(caption2, shift=UP * 0.35), run_time=1.2)
            self.remove(caption)
        self.read_pause()

        # ================= 5. Untrained: nearly flat =================
        uniform_h = PROB_BASE + PROB_SCALE / N
        uniform = DashedLine([CHART_LEFT - 0.05, uniform_h, 0], [CHART_LEFT + PITCH * N + 0.05, uniform_h, 0],
                             color=style.TEXT, stroke_width=2, dash_length=0.08)
        uniform_label = self.text(f"1/{N}", size=style.LABEL_SIZE, color=style.TEXT)
        uniform_label.next_to(uniform, RIGHT, buff=0.15)

        with self.voiceover(text='Before training, the model has no idea. <bookmark mark="flat"/>Every bar '
                                 'is about the same, close to one in sixty-five.'):
            self.wait_until_bookmark("flat")
            self.play(Create(uniform), FadeIn(uniform_label, shift=LEFT * 0.15), run_time=0.8)
        self.read_pause()

        # ================= 6. The right answer, in yellow =================
        p_t = PROBS[TARGET]
        pct = f"{p_t * 100:.1f}"
        tbar = logit_bars[TARGET]
        t_char = self.char_label(VOCAB[TARGET], size=style.BODY_SIZE, color=style.YELLOW)
        t_char.next_to([bar_x(TARGET), PROB_BASE, 0], DOWN, buff=0.15)
        t_p = self.text(f"p = {p_t:.3f}", size=style.LABEL_SIZE, color=style.YELLOW)
        t_p.move_to([bar_x(TARGET) + 0.9, PROB_BASE + 2.15, 0])
        t_arrow = Arrow(t_p.get_bottom() + LEFT * 0.4, tbar.get_top(), buff=0.08, color=style.YELLOW,
                        stroke_width=3, max_tip_length_to_length_ratio=0.25)
        ghost = self.small_token(" ", color=style.YELLOW).next_to(tokens, RIGHT, buff=0.1)

        with self.voiceover(text=f'The right answer here is a <bookmark mark="sp"/>space, and it gets '
                                 f'just <bookmark mark="p"/>{pct} percent.'):
            self.wait_until_bookmark("sp")
            self.play(FadeIn(ghost, shift=LEFT * 0.15), tbar.animate.set_fill(style.YELLOW, opacity=1),
                      FadeIn(t_char, shift=UP * 0.1), run_time=0.6)
            self.wait_until_bookmark("p")
            self.play(GrowArrow(t_arrow), FadeIn(t_p), run_time=0.6)
        self.read_pause(3)

        # ================= 7. Hand-off to line 7 =================
        ring = SurroundingRectangle(VGroup(t_p, tbar), color=style.RED, buff=0.1, corner_radius=0.08)
        with self.voiceover(text='How bad is that guess? <bookmark mark="loss"/>Line seven puts a number on it.'):
            self.wait_until_bookmark("loss")
            self.play(Create(ring), panel.focus(7), run_time=0.7)
        self.read_pause(3)

        self.fade_out_all()
