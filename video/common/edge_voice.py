"""manim-voiceover SpeechService for Microsoft Edge neural voices (free, natural-sounding).

Needs internet the first time a sentence is spoken; after that the mp3 is cached per scene.
"""

import asyncio
from pathlib import Path

import edge_tts
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import SpeechService, path_to_string


class EdgeTTSService(SpeechService):
    def __init__(self, voice: str, rate: str = "+0%", pitch: str = "+0Hz", **kwargs):
        super().__init__(**kwargs)
        self.voice, self.rate, self.pitch = voice, rate, pitch

    def generate_from_text(self, text, cache_dir=None, path=None, **kwargs):
        cache_dir = cache_dir or self.cache_dir
        input_text = remove_bookmarks(text)
        input_data = {"input_text": input_text, "service": "edge-tts",
                      "voice": self.voice, "rate": self.rate, "pitch": self.pitch}
        cached = self.get_cached_result(input_data, cache_dir)
        if cached is not None:
            return cached

        audio_path = path_to_string(path) if path else self.get_audio_basename(input_data) + ".mp3"
        communicate = edge_tts.Communicate(input_text, self.voice, rate=self.rate, pitch=self.pitch)
        asyncio.run(communicate.save(str(Path(cache_dir) / audio_path)))
        return {"input_text": text, "input_data": input_data, "original_audio": audio_path}
