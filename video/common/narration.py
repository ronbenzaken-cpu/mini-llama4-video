"""NarratedScene: the base class of every scene. The voice and standard helpers live here.

See video/scenes/s0_0_base_example.py for a complete example of how to use it.
"""

from pathlib import Path

from manim import FadeIn, FadeOut, Text, Write
from manim_voiceover import VoiceoverScene

from . import style
from .edge_voice import EdgeTTSService

VOICE_CACHE_ROOT = Path(__file__).resolve().parents[2] / "media" / "voiceovers"


class NarratedScene(VoiceoverScene):
    def setup(self):
        super().setup()
        self.camera.background_color = style.BG
        # One voice cache folder per scene class, so scenes rendered in parallel never
        # write to the same cache.json.
        cache_dir = VOICE_CACHE_ROOT / type(self).__name__
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.set_speech_service(
            EdgeTTSService(voice=style.VOICE, rate=style.VOICE_RATE, cache_dir=str(cache_dir))
        )

    def voiceover(self, *args, **kwargs):
        """Same as VoiceoverScene.voiceover, plus a short silent lead-in before the scene's
        first sentence so players / headphones never clip the first word."""
        if not getattr(self, "_lead_in_done", False):
            self._lead_in_done = True
            self.wait(style.LEAD_IN)
        return super().voiceover(*args, **kwargs)

    # ---------- standard text ----------
    def text(self, s: str, size: float = style.BODY_SIZE, color=style.TEXT, **kw) -> Text:
        return Text(s, font=style.FONT, font_size=size, color=color, **kw)

    def heading(self, s: str, color=style.TEXT) -> Text:
        """Create the scene heading at the standard top position (not yet animated)."""
        return self.text(s, size=style.HEADING_SIZE, color=color).move_to(style.HEADING_POS)

    # ---------- standard beats ----------
    def show_heading(self, s: str, color=style.TEXT) -> Text:
        h = self.heading(s, color)
        self.play(Write(h), run_time=0.8)
        return h

    def read_pause(self, factor: float = 1.0):
        """Give the viewer time to read before the next idea starts."""
        self.wait(style.READ_PAUSE * factor)

    def fade_out_all(self):
        """Standard scene ending: fade everything to the empty background."""
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=style.FADE_OUT)

    def fade_in_all(self, *mobjects, run_time: float = 0.6):
        self.play(*[FadeIn(m) for m in mobjects], run_time=run_time)
