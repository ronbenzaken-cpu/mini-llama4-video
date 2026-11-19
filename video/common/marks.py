"""Check / cross marks drawn as shapes (Consolas has no ✓ / ✗ glyphs)."""

from manim import VMobject, VGroup, Line, UL, UR, DL, DR, LEFT, DOWN, RIGHT, UP

from . import style


def check_mark(size: float = 0.4, color=style.GREEN, stroke: float = 6) -> VMobject:
    m = VMobject(color=color, stroke_width=stroke)
    m.set_points_as_corners([LEFT * 0.5 * size, DOWN * 0.45 * size + LEFT * 0.1 * size,
                             RIGHT * 0.55 * size + UP * 0.5 * size])
    return m


def cross_mark(size: float = 0.4, color=style.RED, stroke: float = 6) -> VGroup:
    s = size / 2
    return VGroup(Line(UL * s, DR * s, color=color, stroke_width=stroke),
                  Line(UR * s, DL * s, color=color, stroke_width=stroke))
