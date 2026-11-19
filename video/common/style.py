"""Shared visual style for every scene (3Blue1Brown-like). Import, don't copy."""

from manim import config

# ---- colors ----
BG = "#1C1C1C"          # background
TEXT = "#ECECEC"        # normal text
DIM = "#7A7A7A"         # secondary / de-emphasized
BLUE = "#58C4DD"        # data, vectors, tokens
YELLOW = "#FFD54F"      # highlight, "look here"
RED = "#FC6255"         # loss, errors, gradients
GREEN = "#83C167"       # updates, correct answers
TEAL = "#5CD0B3"        # secondary data
PURPLE = "#9A72AC"      # experts / routing
GREY_BOX = "#333333"    # box fills

# ---- type ----
FONT = "Consolas"       # monospace looks good for both code and labels on Windows
TITLE_SIZE = 56
HEADING_SIZE = 40
BODY_SIZE = 30
LABEL_SIZE = 22
CODE_SIZE = 20

# ---- timing ----
FADE_OUT = 0.5          # every scene ends by fading everything out over this time

config.background_color = BG
