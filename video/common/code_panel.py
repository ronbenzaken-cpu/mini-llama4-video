"""CodePanel: the 9-line training step shown on the left 40% of every Act 3 scene.

    panel = CodePanel()                       # positioned at style.CODE_PANEL_CENTER
    self.play(FadeIn(panel))
    self.play(panel.focus(4))                 # highlight line 4, dim the others (returns an animation)
    panel.mark_done(1, 2, 3)                  # dim check-state for finished lines (instant)
"""

from manim import (LEFT, AnimationGroup, Code, RoundedRectangle, VGroup)

from . import style

HERO_CODE = """\
x, y   = batch[:, :-1], batch[:, 1:]
h      = model.tok_embeddings(x)
for layer in model.layers:
    h = h + layer.attn(layer.attn_norm(h))
    h = h + layer.ffn(layer.ffn_norm(h))
logits = model.output(model.norm(h))
loss   = cross_entropy(logits, y)
loss.backward()
opt.step(); opt.zero_grad()"""

N_LINES = 9
PANEL_WIDTH = 5.4   # fits the left 40% of the frame


class CodePanel(VGroup):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.code = Code(code_string=HERO_CODE, language="python", formatter_style="monokai",
                         add_line_numbers=True, background="rectangle",
                         background_config={"fill_color": style.GREY_BOX, "stroke_width": 0,
                                            "corner_radius": 0.15},
                         paragraph_config={"font": style.FONT})
        self.code.scale_to_fit_width(PANEL_WIDTH)
        self.lines = self.code.code_lines          # Paragraph: one submobject per line
        self.numbers = self.code.submobjects[1] if len(self.code.submobjects) > 2 else None
        self.highlight = RoundedRectangle(corner_radius=0.06, height=0.1, width=0.1,
                                          stroke_color=style.YELLOW, stroke_width=2.5,
                                          fill_color=style.YELLOW, fill_opacity=0.12)
        self.highlight.set_opacity(0)
        self.add(self.code, self.highlight)
        self.move_to(style.CODE_PANEL_CENTER)

    def line(self, n: int):
        """1-based line n of the hero code."""
        return self.lines[n - 1]

    def _box_for(self, n: int) -> RoundedRectangle:
        ln = self.line(n)
        bg = self.code.submobjects[0]
        box = RoundedRectangle(corner_radius=0.06, width=bg.width - 0.15, height=ln.height + 0.08,
                               stroke_color=style.YELLOW, stroke_width=2.5,
                               fill_color=style.YELLOW, fill_opacity=0.12)
        box.move_to(ln).align_to(bg, LEFT).shift([0.075, 0, 0])
        return box

    def focus(self, n: int, dim: float = 0.35):
        """Animation: highlight line n and dim the other lines."""
        anims = [self.highlight.animate.become(self._box_for(n))]
        for i in range(1, N_LINES + 1):
            anims.append(self.line(i).animate.set_opacity(1.0 if i == n else dim))
        return AnimationGroup(*anims)

    def focus_now(self, n: int, dim: float = 0.35):
        """Same as focus(), applied instantly (use before the panel first appears)."""
        self.highlight.become(self._box_for(n))
        for i in range(1, N_LINES + 1):
            self.line(i).set_opacity(1.0 if i == n else dim)
        return self
