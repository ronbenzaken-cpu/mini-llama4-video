"""manim-voiceover SpeechService for Microsoft Edge neural voices (free, natural-sounding).

Each clip is trimmed of the silence Edge puts at its start/end, so sentences follow each
other tightly; the gap between sentences is controlled only by style.READ_PAUSE.
Needs internet the first time a sentence is spoken; after that the mp3 is cached per scene.
"""

import asyncio
from pathlib import Path

import av
import edge_tts
import numpy as np
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import SpeechService, path_to_string

KEEP_MS = 40          # silence kept at each end after trimming
THRESHOLD = 0.01      # amplitude below this counts as silence


def trim_silence(path: Path) -> None:
    with av.open(str(path)) as c:
        sr = c.streams.audio[0].rate
        s = np.concatenate([f.to_ndarray().mean(0) for f in c.decode(audio=0)]).astype(np.float32)
    loud = np.flatnonzero(np.abs(s) > THRESHOLD)
    if loud.size == 0:
        return
    keep = int(sr * KEEP_MS / 1000)
    s = s[max(loud[0] - keep, 0): loud[-1] + keep]
    with av.open(str(path), "w") as out:
        st = out.add_stream("mp3", rate=sr)
        st.layout = "mono"
        frame = av.AudioFrame.from_ndarray(s[None, :], format="flt", layout="mono")
        frame.sample_rate = sr
        for p in st.encode(frame):
            out.mux(p)
        for p in st.encode(None):
            out.mux(p)


class EdgeTTSService(SpeechService):
    def __init__(self, voice: str, rate: str = "+0%", pitch: str = "+0Hz", **kwargs):
        super().__init__(**kwargs)
        self.voice, self.rate, self.pitch = voice, rate, pitch

    def generate_from_text(self, text, cache_dir=None, path=None, **kwargs):
        cache_dir = cache_dir or self.cache_dir
        input_text = remove_bookmarks(text)
        input_data = {"input_text": input_text, "service": "edge-tts", "trim": KEEP_MS,
                      "voice": self.voice, "rate": self.rate, "pitch": self.pitch}
        cached = self.get_cached_result(input_data, cache_dir)
        if cached is not None:
            return cached

        audio_path = path_to_string(path) if path else self.get_audio_basename(input_data) + ".mp3"
        full = Path(cache_dir) / audio_path
        asyncio.run(edge_tts.Communicate(input_text, self.voice, rate=self.rate, pitch=self.pitch).save(str(full)))
        trim_silence(full)
        return {"input_text": text, "input_data": input_data, "original_audio": audio_path}
