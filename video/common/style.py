"""Shared visual style + voice for every scene (3Blue1Brown-like). Import, don't copy."""

from manim import DOWN, LEFT, RIGHT, UP, config

# ---- colors (use them by MEANING, not by taste) ----
BG = "#1C1C1C"          # background
TEXT = "#ECECEC"        # normal text
DIM = "#7A7A7A"         # secondary / de-emphasized / footnotes
BLUE = "#58C4DD"        # data: tokens, vectors, the model
YELLOW = "#FFD54F"      # "look here": the current focus / highlighted code line
RED = "#FC6255"         # loss, errors, gradients
GREEN = "#83C167"       # correct answers, weight updates
TEAL = "#5CD0B3"        # secondary data
PURPLE = "#9A72AC"      # experts / routing
GREY_BOX = "#333333"    # box fills

# ---- type ----
FONT = "Consolas"       # monospace; lacks ✓ ✗ ∝ -> see marks.py / SCRIPT.md
MATH_FONT = "Cambria Math"
TITLE_SIZE = 56         # scene title card
HEADING_SIZE = 36       # the heading at the top of a scene
BODY_SIZE = 30          # main labels
LABEL_SIZE = 24         # small labels (minimum for anything the viewer must read)
FOOTNOTE_SIZE = 20      # dim side notes only
CODE_SIZE = 20

# ---- layout (frame is 14.2 x 8 units, origin at the center) ----
SAFE_W, SAFE_H = 13.0, 7.2                  # keep everything inside this box
HEADING_POS = UP * 3.3                      # where self.heading() puts the scene heading
CONTENT_CENTER = DOWN * 0.3                 # center of the area below the heading
# Act 3 split screen: code on the left 40%, animation on the right 60%
CODE_PANEL_CENTER = LEFT * 4.4
ANIM_CENTER = RIGHT * 2.6

# ---- timing ----
READ_PAUSE = 0.6        # pause after each narration block so labels can be read
LEAD_IN = 0.3           # silent pause before the first sentence, so the first word is never clipped
FADE_OUT = 0.5          # every scene ends by fading everything out over this time

# ---- voice (Microsoft Edge neural TTS) ----
VOICE = "en-GB-RyanNeural"         # chosen by the user from renders/voice-samples
VOICE_RATE = "+8%"      # slightly faster than default, sounds more natural

config.background_color = BG
