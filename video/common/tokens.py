"""Token boxes and baseline-safe text."""

from manim import RIGHT, RoundedRectangle, Text, VGroup

from . import style


def word(s: str, size: float = style.BODY_SIZE, color=style.TEXT) -> Text:
    """Text that sits on a common baseline no matter which letters it has.

    Manim centers Text by its glyph shapes, so "or" floats higher than "To" or "py".
    We wrap the word in invisible '|' struts (they span ascender to descender), which
    gives every word the same vertical extent.
    """
    t = Text(f"|{s}|", font=style.FONT, font_size=size, color=color)
    t[0].set_opacity(0)
    t[-1].set_opacity(0)
    return t


def token_box(s: str, color=style.BLUE, size: float = style.BODY_SIZE, height: float = 0.8) -> VGroup:
    """A rounded box with a word inside: the standard look of a token."""
    label = word(s, size=size)
    box = RoundedRectangle(corner_radius=0.12, width=max(label.width + 0.2, 0.9), height=height,
                           stroke_color=color, fill_color=style.GREY_BOX, fill_opacity=1)
    label.move_to(box)
    return VGroup(box, label)


def token_row(words, color=style.BLUE, buff: float = 0.2, **kw) -> VGroup:
    return VGroup(*[token_box(w, color=color, **kw) for w in words]).arrange(RIGHT, buff=buff)
