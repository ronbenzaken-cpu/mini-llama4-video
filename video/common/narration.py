"""NarratedScene: the one place the voice is configured.

Usage in a scene:

    from video.common.narration import NarratedScene

    class Scene_0_1(NarratedScene):
        def construct(self):
            with self.voiceover(text="Hello world.") as tracker:
                self.play(Write(title), run_time=tracker.duration)
            self.fade_out_all()
"""

from manim import FadeOut
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

from pathlib import Path

from . import style

VOICE_CACHE_ROOT = Path(__file__).resolve().parents[2] / "media" / "voiceovers"


class NarratedScene(VoiceoverScene):
    def setup(self):
        super().setup()
        self.camera.background_color = style.BG
        # One voice cache folder per scene class, so scenes rendered in parallel by different
        # agents never write to the same cache.json (shared cache got corrupted otherwise).
        cache_dir = VOICE_CACHE_ROOT / type(self).__name__
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.set_speech_service(GTTSService(lang="en", tld="com", cache_dir=str(cache_dir)))

    def fade_out_all(self):
        """Standard scene ending: fade everything to the empty background."""
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=style.FADE_OUT)
