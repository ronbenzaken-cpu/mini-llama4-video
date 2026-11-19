"""Scene 1.6 - Mixture of Experts.

Goal: a router sends each token to 1 of 128 experts, plus a shared expert that every token uses.
Facts: docs/llama4_architecture.md section 2.5
  * shared expert for every token + N routed experts (Maverick: 128)
  * router scores the experts, token goes to the top-1 expert
  * the router's sigmoid score scales the expert's input:  out = Shared(x) + Expert_top1(x * sigma(score))
  * 400B parameters stored, 17B used per token
Which expert each word picks is illustrative (Llama 4's real routing is not published); no gate numbers
are invented, sigma is shown as a symbol.

Preview:  .venv/Scripts/python -m manim -ql video/scenes/s1_6_moe.py Scene_1_6
Final:    .venv/Scripts/python -m manim -qh video/scenes/s1_6_moe.py Scene_1_6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manim import (DOWN, LEFT, RIGHT, UP, Create, FadeIn, FadeOut, Indicate, Line,  # noqa: E402
                   ReplacementTransform, RoundedRectangle, Square, SurroundingRectangle, Transform, ValueTracker,
                   VGroup, Write, always_redraw, smooth)

from video.common import style  # noqa: E402
from video.common.narration import NarratedScene  # noqa: E402
from video.common.tokens import token_box  # noqa: E402

N_EXPERTS, ROWS, COLS = 128, 8, 16        # Maverick: 128 routed experts (doc section 2.5)
STORED_B, ACTIVE_B = 400, 17              # billions of parameters (doc section 2.5)
CELL = 0.34
GRID_CENTER = RIGHT * 1.3 + DOWN * 0.55
SHARED_CENTER = RIGHT * 1.3 + UP * 1.85
ROUTER_CENTER = LEFT * 3.7 + UP * 0.1
TOKEN_START = LEFT * 6.0 + UP * 0.1
# (word, routed expert index): illustrative choices, spread over the grid
ROUTES = [("To", 37), ("be", 90), ("or", 6), ("not", 121)]


class Scene_1_6(NarratedScene):
    # ---------- builders ----------
    def box(self, label: str, width: float, height: float, color, size=style.BODY_SIZE) -> VGroup:
        rect = RoundedRectangle(corner_radius=0.12, width=width, height=height, stroke_color=color,
                                stroke_width=3, fill_color=color, fill_opacity=0.15)
        txt = self.text(label, size=size, color=style.TEXT).move_to(rect)
        return VGroup(rect, txt)

    def expert_grid(self) -> VGroup:
        return VGroup(*[
            Square(side_length=CELL * 0.78, stroke_width=1.5, stroke_color=style.PURPLE,
                   fill_color=style.PURPLE, fill_opacity=0.3)
            for _ in range(N_EXPERTS)
        ]).arrange_in_grid(rows=ROWS, cols=COLS, buff=CELL * 0.22).move_to(GRID_CENTER)

    def route_one(self, word: str, idx: int, grid, router, shared, speed: float = 1.0, gate_note=None):
        """One token: enters, goes into the router, lights the shared expert + one routed expert."""
        tok = token_box(word, size=style.LABEL_SIZE, height=0.6).move_to(TOKEN_START)
        cell = grid[idx]
        a_shared = Line(router.get_right(), shared.get_left(), color=style.YELLOW, stroke_width=4)
        a_exp = Line(router.get_right(), cell.get_center(), color=style.YELLOW, stroke_width=4, buff=0.0)
        gate = self.text("× σ", size=style.BODY_SIZE, color=style.YELLOW)
        gate.move_to(a_exp.point_from_proportion(0.3)).shift(DOWN * 0.35 if idx >= 64 else UP * 0.35)

        self.play(FadeIn(tok, shift=RIGHT * 0.4), run_time=0.4 * speed)
        self.play(tok.animate.move_to(router).scale(0.5).set_opacity(0),
                  Indicate(router, color=style.YELLOW, scale_factor=1.08), run_time=0.5 * speed)
        self.remove(tok)
        self.play(Create(a_shared), shared[0].animate.set_fill(style.PURPLE, opacity=0.55),
                  run_time=0.4 * speed)
        self.play(Create(a_exp), FadeIn(gate, scale=0.6),
                  cell.animate.set_fill(style.YELLOW, opacity=1).set_stroke(style.YELLOW, width=3),
                  run_time=0.5 * speed)
        return VGroup(a_shared, a_exp, gate), cell

    def clear_route(self, lines, cell, shared, speed: float = 1.0):
        self.play(FadeOut(lines),
                  cell.animate.set_fill(style.PURPLE, opacity=0.3).set_stroke(style.PURPLE, width=1.5),
                  shared[0].animate.set_fill(style.PURPLE, opacity=0.15), run_time=0.3 * speed)

    # ---------- the scene ----------
    def construct(self):
        # ================= 1. What this scene is about: the feed-forward half =================
        heading = self.heading("Mixture of Experts")
        self.play(Write(heading), run_time=style.LEAD_IN)

        ffn = self.box("feed-forward network", 5.4, 2.7, style.BLUE)
        ffn.move_to(GRID_CENTER)
        ffn_tag = self.text("one big FFN", size=style.LABEL_SIZE, color=style.DIM)
        ffn_tag.next_to(ffn, DOWN, buff=0.3)
        with self.voiceover(text='We\'ve seen attention. Now the <bookmark mark="ffn"/>other half of each block: '
                                 'the feed-forward network.'):
            self.wait_until_bookmark("ffn")
            self.play(FadeIn(ffn, shift=UP * 0.2), FadeIn(ffn_tag), run_time=0.8)
        self.read_pause()

        # ================= 2. Split it into 128 experts =================
        grid = self.expert_grid()
        grid_tag = self.text("128 routed experts", size=style.LABEL_SIZE, color=style.PURPLE)
        grid_tag.next_to(grid, DOWN, buff=0.3)
        with self.voiceover(text='Instead of one big network, Llama 4 Maverick '
                                 '<bookmark mark="split"/>splits it into 128 small experts.'):
            self.wait_until_bookmark("split")
            self.play(ReplacementTransform(ffn[0], grid), FadeOut(ffn[1], scale=0.5),
                      Transform(ffn_tag, grid_tag),
                      run_time=1.4, rate_func=smooth)
        self.read_pause()

        # ================= 3. The router =================
        router = self.box("router", 1.6, 0.9, style.PURPLE)
        router.move_to(ROUTER_CENTER)
        router_note = self.text("scores all 128", size=style.LABEL_SIZE, color=style.DIM)
        router_note.next_to(router, DOWN, buff=0.25)
        with self.voiceover(text='A small <bookmark mark="router"/>router looks at each token '
                                 'and <bookmark mark="pick"/>picks just one of them.'):
            self.wait_until_bookmark("router")
            self.play(FadeIn(router, shift=RIGHT * 0.3), run_time=0.6)
            self.wait_until_bookmark("pick")
            self.play(FadeIn(router_note, shift=UP * 0.15), run_time=0.5)
        self.read_pause()

        # ================= 4. The shared expert =================
        shared = self.box("shared expert", 5.4, 0.7, style.PURPLE, size=style.LABEL_SIZE)
        shared.move_to(SHARED_CENTER)
        shared_note = self.text("every token", size=style.LABEL_SIZE, color=style.DIM)
        shared_note.next_to(shared, RIGHT, buff=0.25)
        with self.voiceover(text='There is also <bookmark mark="shared"/>one shared expert '
                                 'that <bookmark mark="every"/>every token uses.'):
            self.wait_until_bookmark("shared")
            self.play(FadeIn(shared, shift=DOWN * 0.2), run_time=0.6)
            self.wait_until_bookmark("every")
            self.play(FadeIn(shared_note, shift=LEFT * 0.15), run_time=0.5)
        self.read_pause()

        # ================= 5. First token, slowly, with the gate =================
        word, idx = ROUTES[0]
        with self.voiceover(text='Watch a token. It <bookmark mark="in"/>enters the router, '
                                 'goes through <bookmark mark="sh"/>the shared expert, '
                                 'and through <bookmark mark="ex"/>the one expert the router chose.'):
            self.wait_until_bookmark("in")
            tok = token_box(word, size=style.LABEL_SIZE, height=0.6).move_to(TOKEN_START)
            self.play(FadeIn(tok, shift=RIGHT * 0.4), run_time=0.4)
            self.play(tok.animate.move_to(router).scale(0.5).set_opacity(0),
                      Indicate(router, color=style.YELLOW, scale_factor=1.08), run_time=0.6)
            self.remove(tok)
            a_shared = Line(router.get_right(), shared.get_left(), color=style.YELLOW, stroke_width=4)
            cell = grid[idx]
            a_exp = Line(router.get_right(), cell.get_center(), color=style.YELLOW, stroke_width=4)
            self.wait_until_bookmark("sh")
            self.play(Create(a_shared), shared[0].animate.set_fill(style.PURPLE, opacity=0.55), run_time=0.5)
            self.wait_until_bookmark("ex")
            self.play(Create(a_exp),
                      cell.animate.set_fill(style.YELLOW, opacity=1).set_stroke(style.YELLOW, width=3),
                      run_time=0.6)
        self.read_pause()

        gate = self.text("× σ", size=style.BODY_SIZE, color=style.YELLOW)
        gate.move_to(a_exp.point_from_proportion(0.3)).shift(UP * 0.35)
        gate_note = VGroup(self.text("σ = sigmoid of", size=style.LABEL_SIZE, color=style.YELLOW),
                           self.text("the router's score", size=style.LABEL_SIZE, color=style.YELLOW)
                           ).arrange(DOWN, buff=0.12)
        gate_note.next_to(router_note, DOWN, buff=0.35)
        with self.voiceover(text='The router\'s score, squashed by a <bookmark mark="sig"/>sigmoid, '
                                 '<bookmark mark="scale"/>scales how strongly the token goes into that expert.'):
            self.wait_until_bookmark("sig")
            self.play(FadeIn(gate_note, shift=UP * 0.15), run_time=0.6)
            self.wait_until_bookmark("scale")
            self.play(FadeIn(gate, scale=0.6), Indicate(cell, color=style.YELLOW, scale_factor=1.6), run_time=0.8)
        self.read_pause(3)
        self.clear_route(VGroup(a_shared, a_exp, gate), cell, shared)

        # ================= 6. More tokens, different experts =================
        with self.voiceover(text='Each <bookmark mark="next"/>new token may pick a different expert; '
                                 'the other 127 stay idle.'):
            self.wait_until_bookmark("next")
            for word, idx in ROUTES[1:]:
                lines, cell = self.route_one(word, idx, grid, router, shared, speed=0.75)
                self.wait(0.25)
                self.clear_route(lines, cell, shared, speed=0.75)
        self.read_pause()

        # ================= 7. Stored vs used =================
        stored = ValueTracker(0)
        used = ValueTracker(0)
        counter = always_redraw(lambda: VGroup(
            self.text(f"{int(round(stored.get_value()))}B stored", size=style.BODY_SIZE, color=style.PURPLE),
            self.text("/", size=style.BODY_SIZE, color=style.DIM),
            self.text(f"{int(round(used.get_value()))}B used per token", size=style.BODY_SIZE,
                      color=style.YELLOW),
        ).arrange(RIGHT, buff=0.3).move_to(DOWN * 3.15))
        with self.voiceover(text='So the model <bookmark mark="st"/>stores 400 billion parameters, '
                                 'but each token <bookmark mark="us"/>uses only 17 billion.'):
            self.add(counter)
            self.wait_until_bookmark("st")
            self.play(stored.animate.set_value(STORED_B), grid.animate.set_fill(opacity=0.55),
                      run_time=1.0, rate_func=smooth)
            self.wait_until_bookmark("us")
            lines, cell = self.route_one("to", 52, grid, router, shared, speed=0.6)
            self.play(used.animate.set_value(ACTIVE_B), run_time=0.8, rate_func=smooth)
        self.read_pause()

        with self.voiceover(text='<bookmark mark="k"/>Huge knowledge, small cost per token.'):
            self.wait_until_bookmark("k")
            frame = SurroundingRectangle(counter, color=style.YELLOW, buff=0.15, corner_radius=0.1)
            self.play(Create(frame), run_time=0.8)
        self.read_pause()
        with self.voiceover(text='Next, we shrink this whole design down to laptop size.'):
            pass
        self.read_pause(2)

        self.fade_out_all()
