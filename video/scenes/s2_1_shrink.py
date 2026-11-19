"""Scene 2.1 - Shrinking it: the same Llama 4 design, laptop-sized numbers.

A small pipeline (tokens -> embedding -> block -> next-token scores) is shown with the real
Llama 4 Maverick sizes. Then, one number at a time and on the spoken word, each size counts
down to our mini model's value (log-scale countdown, so it visibly "shrinks"):
vocab 202,048 -> 65, width 5,120 -> 128, layers 48 -> 4, experts 128 -> 4, params 400B -> 1.4M.
Finally a laptop outline is drawn around the diagram and the heading morphs to "Mini Llama 4".

Llama 4 sizes: docs/llama4_architecture.md (sections 1, 2, 4). Mini sizes: TRACE["config"].

Preview:  .venv/Scripts/python -m manim -ql video/scenes/s2_1_shrink.py Scene_2_1
Final:    .venv/Scripts/python -m manim -qh video/scenes/s2_1_shrink.py Scene_2_1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, FadeIn, GrowArrow,  # noqa: E402
                   LaggedStart, Polygon, ReplacementTransform, RoundedRectangle, Square,
                   TransformMatchingShapes, ValueTracker, VGroup,
                   Write, always_redraw, smooth)

from video.common import style  # noqa: E402
from video.common.data import TRACE  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402

CFG = TRACE["config"]

# Real Llama 4 Maverick sizes (docs/llama4_architecture.md)
LLAMA_VOCAB, LLAMA_DIM, LLAMA_LAYERS, LLAMA_EXPERTS, LLAMA_PARAMS = 202_048, 5_120, 48, 128, 400e9

BOX_Y = 0.9          # centre line of the pipeline boxes
COUNT_TIME = 1.1     # how long one countdown takes


def fmt_int(v: float) -> str:
    return f"{int(round(v)):,}"


def fmt_params(v: float) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= scale:
            x = v / scale
            return f"{x:.0f}{unit}" if x >= 10 else f"{x:.1f}{unit}"
    return f"{v:.0f}"


class Scene_2_1(NarratedScene):
    # ---------------- builders ----------------
    def stage(self, name: str, width: float, height: float = 1.1, color=style.BLUE) -> VGroup:
        box = RoundedRectangle(corner_radius=0.15, width=width, height=height, stroke_color=color,
                               fill_color=style.GREY_BOX, fill_opacity=1)
        label = self.text(name, size=style.LABEL_SIZE, color=style.TEXT).move_to(box).set_z_index(2)
        return VGroup(box, label)

    def stat(self, name: str, big: float, small: float, fmt, anchor):
        """A stat = dim name + a live number that counts from `big` down to `small` (log scale)."""
        t = ValueTracker(0.0)
        state = {"color": style.BLUE}
        title = self.text(name, size=style.LABEL_SIZE, color=style.DIM).move_to(anchor)
        value = always_redraw(lambda: self.text(fmt(big * (small / big) ** t.get_value()),
                                                size=style.BODY_SIZE, color=state["color"])
                              .next_to(title, DOWN, buff=0.15))
        return {"title": title, "value": value, "t": t, "state": state, "fmt": fmt,
                "big": big, "small": small}

    def count_down(self, s, extra_anims=()):
        """Count one stat down (yellow while moving), then freeze it and show the old size."""
        s["state"]["color"] = style.YELLOW
        self.play(s["t"].animate.set_value(1.0), *extra_anims, run_time=COUNT_TIME, rate_func=smooth)
        static = self.text(s["fmt"](s["small"]), size=style.BODY_SIZE, color=style.BLUE).move_to(s["value"])
        self.remove(s["value"])
        self.add(static)
        s["value"] = static
        was = self.text("was " + s["fmt"](s["big"]), size=style.LABEL_SIZE, color=style.DIM)
        was.next_to(static, DOWN, buff=0.12)
        s["was"] = was
        self.play(FadeIn(was, shift=DOWN * 0.15), run_time=0.4)

    def expert_grid(self, n: int, side: float, buff: float, cols: int) -> VGroup:
        sq = VGroup(*[Square(side_length=side, stroke_width=1 if side < 0.2 else 2,
                             stroke_color=style.PURPLE, fill_color=style.PURPLE, fill_opacity=0.35)
                      for _ in range(n)])
        return sq.arrange_in_grid(cols=cols, buff=buff).set_z_index(2)

    def laptop(self, screen_w: float, screen_h: float) -> VGroup:
        screen = RoundedRectangle(corner_radius=0.2, width=screen_w, height=screen_h,
                                  stroke_color=style.TEXT, stroke_width=4)
        bw, bh = screen_w + 1.0, 0.35
        top = screen.get_bottom() + DOWN * 0.08
        base = Polygon(top + LEFT * screen_w / 2, top + RIGHT * screen_w / 2,
                       top + RIGHT * bw / 2 + DOWN * bh, top + LEFT * bw / 2 + DOWN * bh,
                       stroke_color=style.TEXT, stroke_width=4, fill_color=style.GREY_BOX, fill_opacity=1)
        return VGroup(screen, base)

    # ---------------- scene ----------------
    def construct(self):
        heading = self.heading("Shrinking Llama 4")

        # --- the pipeline: tokens -> embedding -> block (attention + experts) -> next-token scores ---
        tokens = self.stage("tokens", 2.0)
        embed = self.stage("embedding", 2.4)
        block_box = RoundedRectangle(corner_radius=0.15, width=3.6, height=2.0, stroke_color=style.BLUE,
                                     fill_color=style.GREY_BOX, fill_opacity=1)
        block_lbl = self.text("block", size=style.LABEL_SIZE).set_z_index(2)
        out = self.stage("next-token\nscores", 2.4)
        row = VGroup(tokens, embed, block_box, out).arrange(RIGHT, buff=0.75).move_to(UP * BOX_Y)
        block_lbl.move_to(block_box.get_top() + DOWN * 0.3)
        # a few ghost copies behind the block: "a stack of blocks"
        ghosts = VGroup(*[block_box.copy().set_fill(opacity=1).set_stroke(style.DIM, width=2)
                          .shift((UP + RIGHT) * 0.1 * k) for k in (2, 1)])
        block = VGroup(ghosts, block_box, block_lbl)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.1, color=style.DIM,
                                max_tip_length_to_length_ratio=0.35)
                          for a, b in ((tokens, embed), (embed, block_box), (block_box, out))])

        grid_big = self.expert_grid(LLAMA_EXPERTS, 0.11, 0.035, 16)
        grid_big.move_to(block_box.get_center() + DOWN * 0.25)
        grid_small = self.expert_grid(CFG["n_experts"], 0.42, 0.18, CFG["n_experts"])
        grid_small.move_to(grid_big)

        # --- the five stats under the pipeline ---
        stat_y = row.get_bottom()[1] - 0.45
        vocab = self.stat("vocabulary", LLAMA_VOCAB, CFG["vocab_size"], fmt_int, [tokens.get_x(), stat_y, 0])
        width = self.stat("width", LLAMA_DIM, CFG["dim"], fmt_int, [embed.get_x(), stat_y, 0])
        layers = self.stat("layers", LLAMA_LAYERS, CFG["n_layers"], fmt_int,
                           [block_box.get_x() - 0.9, stat_y, 0])
        experts = self.stat("experts", LLAMA_EXPERTS, CFG["n_experts"], fmt_int,
                            [block_box.get_x() + 0.9, stat_y, 0])
        params = self.stat("parameters", LLAMA_PARAMS, CFG["n_params"], fmt_params, [0, -2.0, 0])
        stats = [vocab, width, layers, experts, params]

        # ================= lead-in: heading =================
        self.play(Write(heading), run_time=style.LEAD_IN)

        # ================= 1. the real pipeline, real sizes =================
        with self.voiceover(text='We\'ve seen how the real Llama 4 is built. '
                                 'Here is <bookmark mark="pipe"/>its pipeline again, '
                                 'with <bookmark mark="sizes"/>its real sizes.'):
            self.wait_until_bookmark("pipe")
            self.play(LaggedStart(FadeIn(tokens, shift=UP * 0.2), GrowArrow(arrows[0]),
                                  FadeIn(embed, shift=UP * 0.2), GrowArrow(arrows[1]),
                                  FadeIn(block, shift=UP * 0.2), FadeIn(grid_big), GrowArrow(arrows[2]),
                                  FadeIn(out, shift=UP * 0.2), lag_ratio=0.25), run_time=1.6)
            self.wait_until_bookmark("sizes")
            self.play(LaggedStart(*[FadeIn(VGroup(s["title"], s["value"]), shift=UP * 0.15) for s in stats],
                                  lag_ratio=0.2), run_time=1.2)
            # keep the live counters on screen as themselves (always_redraw needs to stay added)
            for s in stats:
                self.add(s["value"])
        self.read_pause(2)

        # ================= 2. same design, smaller numbers =================
        frames = VGroup(tokens[0], embed[0], block_box, out[0])
        with self.voiceover(text='We keep <bookmark mark="same"/>exactly the same design, '
                                 'and <bookmark mark="shrink"/>shrink every number.'):
            self.wait_until_bookmark("same")
            self.play(frames.animate.set_stroke(style.YELLOW), run_time=0.5)
            self.wait_until_bookmark("shrink")
            self.play(frames.animate.set_stroke(style.BLUE), run_time=0.5)
        self.read_pause()

        # ================= 3. one number at a time =================
        with self.voiceover(text='The vocabulary drops from about two hundred thousand tokens '
                                 '<bookmark mark="v"/>to just sixty-five characters.'):
            self.wait_until_bookmark("v")
            self.count_down(vocab)
        self.read_pause()

        with self.voiceover(text='Each token\'s vector goes from five thousand one hundred twenty numbers '
                                 '<bookmark mark="w"/>to one hundred twenty-eight.'):
            self.wait_until_bookmark("w")
            self.count_down(width)
        self.read_pause()

        ghost_fade = ghosts.animate.set_stroke(opacity=0.4)
        with self.voiceover(text='Forty-eight layers <bookmark mark="l"/>become four.'):
            self.wait_until_bookmark("l")
            self.count_down(layers, extra_anims=(ghost_fade,))
        self.read_pause()

        with self.voiceover(text='And a hundred and twenty-eight experts <bookmark mark="e"/>become four.'):
            self.wait_until_bookmark("e")
            self.count_down(experts, extra_anims=(ReplacementTransform(grid_big, grid_small),))
        self.read_pause()

        with self.voiceover(text='In total, <bookmark mark="p"/>one point four million parameters, '
                                 'instead of four hundred billion.'):
            self.wait_until_bookmark("p")
            self.count_down(params)
        self.read_pause(2)

        # ================= 4. it fits on a laptop =================
        diagram = VGroup(tokens, embed, block, grid_small, out, arrows,
                         *[VGroup(s["title"], s["value"], s["was"]) for s in stats])
        target_scale = 0.85
        target_center = DOWN * 0.15
        laptop = self.laptop(diagram.width * target_scale + 0.8, diagram.height * target_scale + 0.8)
        laptop.move_to(target_center + DOWN * 0.18)
        laptop[0].move_to(target_center)
        laptop[1].next_to(laptop[0], DOWN, buff=0.08)
        mini_heading = self.heading("Mini Llama 4", color=style.BLUE)

        with self.voiceover(text='Small enough to <bookmark mark="lap"/>train on a laptop. '
                                 'This is <bookmark mark="mini"/>our Mini Llama 4.'):
            self.wait_until_bookmark("lap")
            self.play(diagram.animate.scale(target_scale).move_to(target_center), run_time=0.7)
            self.play(Create(laptop[0]), FadeIn(laptop[1], shift=UP * 0.1), run_time=0.9)
            self.wait_until_bookmark("mini")
            self.play(TransformMatchingShapes(heading, mini_heading), run_time=0.8)
        self.read_pause(4)

        self.fade_out_all()
