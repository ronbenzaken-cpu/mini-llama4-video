"""Scene 3.3 - Line 3 of the training step: the layer loop.

The vectors h (8 tokens x 128 numbers) pass through the model's 4 layers, one after another.
Each layer gives back an h of the same shape, which is why the layers can be stacked.

Layout (Act 3): CodePanel on the left, line 3 highlighted; on the right a horizontal row of the
4 real layers (TRACE["layers"]: dense/MoE, RoPE/NoPE) and a small packet of colored columns = h,
colored by the real first 8 embedding dims (TRACE["embedding_first8"]).

Preview:  .venv/Scripts/python -m manim -ql video/scenes/s3_3_loop.py Scene_3_3
Final:    .venv/Scripts/python -m manim -qh video/scenes/s3_3_loop.py Scene_3_3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Arrow, FadeIn, Indicate,  # noqa: E402
                   LaggedStart, Rectangle, RoundedRectangle, ValueTracker, VGroup, Write,
                   ManimColor, always_redraw, interpolate_color, smooth)

from video.common import style  # noqa: E402
from video.common.code_panel import CodePanel  # noqa: E402
from video.common.data import TRACE  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402

LAYERS = TRACE["layers"]                 # [{'index','moe','rope'}] x4
EMB = TRACE["embedding_first8"]          # 8 tokens x 8 dims (real values)
N_LAYERS = len(LAYERS)

BLOCK_W, BLOCK_H, BLOCK_GAP = 1.4, 2.6, 0.45
ROW_Y = 0.4                              # center line of the layer row
WAIT_Y = -2.1                            # where h waits below the row (before / after the loop)
LOW = "#1E3A44"                          # dark end of the blue scale for the vector cells


class Scene_3_3(NarratedScene):
    # ---------- builders ----------
    def layer_block(self, layer: dict) -> VGroup:
        moe = layer["moe"]
        color = style.PURPLE if moe else style.BLUE
        box = RoundedRectangle(corner_radius=0.15, width=BLOCK_W, height=BLOCK_H,
                               stroke_color=color, stroke_width=3,
                               fill_color=style.GREY_BOX, fill_opacity=1)
        idx = self.text(str(layer["index"]), size=style.BODY_SIZE, color=style.TEXT)
        idx.move_to(box.get_top() + DOWN * 0.32)
        kind = self.text("MoE" if moe else "dense", size=style.LABEL_SIZE, color=color)
        pos = self.text("RoPE" if layer["rope"] else "NoPE", size=style.LABEL_SIZE,
                        color=style.TEXT if layer["rope"] else style.YELLOW)
        sub = VGroup(kind, pos).arrange(DOWN, buff=0.1).move_to(box.get_bottom() + UP * 0.45)
        return VGroup(box, idx, sub)

    def vector_packet(self) -> VGroup:
        """h as 8 thin columns (one per token), cells colored by the real embedding values."""
        flat = [v for row in EMB for v in row]
        lo, hi = min(flat), max(flat)
        cell = 0.095
        cols = VGroup()
        for tok in EMB:
            col = VGroup(*[Rectangle(width=cell, height=cell, stroke_width=0, fill_opacity=1,
                                     fill_color=interpolate_color(ManimColor(LOW), ManimColor(style.BLUE), (v - lo) / (hi - lo)))
                           for v in tok]).arrange(DOWN, buff=0.012)
            cols.add(col)
        cols.arrange(RIGHT, buff=0.03)
        frame = RoundedRectangle(corner_radius=0.05, width=cols.width + 0.12, height=cols.height + 0.12,
                                 stroke_color=style.BLUE, stroke_width=1.5, fill_opacity=0)
        frame.move_to(cols)
        return VGroup(frame, cols)

    # ---------- scene ----------
    def construct(self):
        heading = self.heading("Line 3: the layer loop")
        panel = CodePanel().focus_now(3, dim=1.0)
        panel.highlight.set_stroke(opacity=0).set_fill(opacity=0)   # appears on "line three"

        blocks = VGroup(*[self.layer_block(l) for l in LAYERS]).arrange(RIGHT, buff=BLOCK_GAP)
        blocks.move_to([style.ANIM_CENTER[0], ROW_Y, 0])
        arrows = VGroup(*[Arrow(blocks[i].get_right(), blocks[i + 1].get_left(), buff=0.05,
                                color=style.DIM, stroke_width=3, max_tip_length_to_length_ratio=0.5)
                          for i in range(N_LAYERS - 1)])
        caption = self.text("model.layers", size=style.LABEL_SIZE, color=style.DIM)
        caption.next_to(blocks, UP, buff=0.3).align_to(blocks, LEFT)

        passed = ValueTracker(0)
        counter = always_redraw(lambda: self.text(f"layers passed: {int(round(passed.get_value()))} / {N_LAYERS}",
                                                  size=style.LABEL_SIZE, color=style.YELLOW)
                                .next_to(blocks, UP, buff=0.3).align_to(blocks, RIGHT))

        packet = self.vector_packet()
        h_label = self.text("h: 8 × 128", size=style.LABEL_SIZE, color=style.BLUE)

        def below(block):  # waiting spot under a block
            return [block.get_center()[0], WAIT_Y, 0]

        def inside(block):  # packet spot inside a block, between index and labels
            return block[0].get_center() + UP * 0.12

        packet.move_to(below(blocks[0]))
        h_label.next_to(packet, DOWN, buff=0.2)

        # ===== lead-in: heading + code panel appear before the first word =====
        self.play(Write(heading), FadeIn(panel), run_time=style.LEAD_IN)

        # ===== 1. what this scene is about: line 3 is a loop over the layers =====
        with self.voiceover(text='Now, <bookmark mark="l3"/>line three: a loop over '
                                 'the model\'s <bookmark mark="four"/>four layers.'):
            self.wait_until_bookmark("l3")
            self.play(panel.focus(3), run_time=0.6)
            self.wait_until_bookmark("four")
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.25) for b in blocks], lag_ratio=0.2),
                      FadeIn(caption), run_time=1.0)
            self.play(LaggedStart(*[FadeIn(a) for a in arrows], lag_ratio=0.3), run_time=0.5)
        self.read_pause()

        # ===== 2. the input: the vectors from line two =====
        with self.voiceover(text='We start with <bookmark mark="h"/>the vectors from line two, '
                                 'one column of numbers for each character.'):
            self.wait_until_bookmark("h")
            self.play(FadeIn(packet, shift=UP * 0.2), FadeIn(h_label, shift=UP * 0.2), run_time=0.7)
            self.play(Indicate(packet[1], color=style.YELLOW, scale_factor=1.15), run_time=0.8)
        self.read_pause()

        # ===== 3. the loop: through layer 0, 1, 2, 3, one after another =====
        def tint(block):
            """Leave a little of the layer's color in h: the layer changed the numbers."""
            c = ManimColor(style.PURPLE if LAYERS[blocks.submobjects.index(block)]["moe"] else style.BLUE)
            return [cell.animate.set_fill(interpolate_color(cell.get_fill_color(), c, 0.15))
                    for col in packet[1] for cell in col]

        def enter(i):
            """Move h into block i (counting the block it just left), light the block."""
            anims = [packet.animate.move_to(inside(blocks[i])),
                     h_label.animate.move_to(blocks[i].get_bottom() + DOWN * 0.38),
                     blocks[i][0].animate.set_stroke(style.YELLOW, width=5)]
            if i > 0:
                anims += [blocks[i - 1][0].animate.set_stroke(
                              style.PURPLE if LAYERS[i - 1]["moe"] else style.BLUE, width=3),
                          passed.animate.set_value(i)]
            self.play(*anims, run_time=0.6, rate_func=smooth)
            self.play(*tint(blocks[i]), run_time=0.35)

        with self.voiceover(text='They pass through <bookmark mark="b0"/>layer zero, '
                                 '<bookmark mark="b1"/>then layer one, <bookmark mark="b2"/>layer two, '
                                 '<bookmark mark="b3"/>and layer three, <bookmark mark="out"/>one after another.'):
            self.wait_until_bookmark("b0")
            self.add(counter)
            enter(0)
            for i in range(1, N_LAYERS):
                self.wait_until_bookmark(f"b{i}")
                enter(i)
            self.wait_until_bookmark("out")
            last = blocks[-1]
            self.play(packet.animate.move_to(below(last)),
                      h_label.animate.move_to([min(last.get_center()[0], 5.4), WAIT_Y, 0]
                                              + DOWN * (packet.height / 2 + 0.2 + h_label.height / 2)),
                      last[0].animate.set_stroke(style.PURPLE if LAYERS[-1]["moe"] else style.BLUE, width=3),
                      passed.animate.set_value(N_LAYERS), run_time=0.7, rate_func=smooth)
        self.read_pause()

        # ===== 4. why it can loop: same shape in, same shape out =====
        with self.voiceover(text='Each layer takes h in, and gives back <bookmark mark="same"/>a new h '
                                 'of exactly the same shape, so the next layer can take it.'):
            self.wait_until_bookmark("same")
            self.play(Indicate(h_label, color=style.YELLOW), run_time=0.9)
        self.read_pause()

        # ===== 5. teaser: the loop body is lines four and five =====
        with self.voiceover(text='Inside every layer, two lines run: <bookmark mark="body"/>four and five. '
                                 'Next, <bookmark mark="attn"/>line four: attention.'):
            self.wait_until_bookmark("body")
            self.play(panel.line(4).animate.set_opacity(1), panel.line(5).animate.set_opacity(1),
                      blocks[0][0].animate.set_stroke(style.YELLOW, width=5), run_time=0.6)
            self.wait_until_bookmark("attn")
            self.play(panel.focus(4), run_time=0.6)
        self.read_pause(3)

        self.fade_out_all()
