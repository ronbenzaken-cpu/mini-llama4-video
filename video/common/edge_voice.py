"""manim-voiceover SpeechService for Microsoft Edge neural voices (free, natural-sounding).

* Word timings: Edge reports when every word is spoken, so scenes can sync animations to
  exact words with bookmarks:

      with self.voiceover(text='You give it <bookmark mark="start"/>the start of a sentence') as t:
          self.wait_until_bookmark("start")
          self.play(...)                      # fires exactly when "the" is spoken

* Each clip is trimmed of Edge's built-in silence, so the gap between sentences is set only
  by style.READ_PAUSE.
Needs internet the first time a sentence is spoken; after that the mp3 is cached per scene.
"""

import asyncio
from pathlib import Path

import av
import edge_tts
import numpy as np
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import SpeechService, path_to_string

TICKS = 10_000_000    # Edge and manim-voiceover both use 100-ns ticks for audio offsets
KEEP_MS = 40          # silence kept at each end after trimming
THRESHOLD = 0.01      # amplitude below this counts as silence


def trim_silence(path: Path) -> float:
    """Trim silence at both ends in place. Returns how many seconds were cut from the start."""
    with av.open(str(path)) as c:
        sr = c.streams.audio[0].rate
        s = np.concatenate([f.to_ndarray().mean(0) for f in c.decode(audio=0)]).astype(np.float32)
    loud = np.flatnonzero(np.abs(s) > THRESHOLD)
    if loud.size == 0:
        return 0.0
    keep = int(sr * KEEP_MS / 1000)
    start = max(loud[0] - keep, 0)
    s = s[start: loud[-1] + keep]
    with av.open(str(path), "w") as out:
        st = out.add_stream("mp3", rate=sr)
        st.layout = "mono"
        frame = av.AudioFrame.from_ndarray(s[None, :], format="flt", layout="mono")
        frame.sample_rate = sr
        for p in st.encode(frame):
            out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    return start / sr


async def _synthesize(text: str, voice: str, rate: str, pitch: str, out: Path):
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary")
    words = []
    with open(out, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append((chunk["offset"], chunk["duration"], chunk["text"]))
    return words


def to_word_boundaries(text: str, words, cut_seconds: float):
    """Convert Edge word events to manim-voiceover's format (text position -> audio time)."""
    boundaries, pos = [], 0
    for offset, duration, w in words:
        found = text.find(w, pos)
        if found < 0:
            continue
        pos = found + len(w)
        boundaries.append({
            "audio_offset": max(int(offset - cut_seconds * TICKS), 0),
            "duration_milliseconds": int(duration / 10_000),
            "text_offset": found,
            "word_length": len(w),
            "text": w,
            "boundary_type": "Word",
        })
    return boundaries


class EdgeTTSService(SpeechService):
    def __init__(self, voice: str, rate: str = "+0%", pitch: str = "+0Hz", **kwargs):
        super().__init__(**kwargs)
        self.voice, self.rate, self.pitch = voice, rate, pitch

    def generate_from_text(self, text, cache_dir=None, path=None, **kwargs):
        cache_dir = cache_dir or self.cache_dir
        input_text = remove_bookmarks(text)
        input_data = {"input_text": input_text, "service": "edge-tts", "trim": KEEP_MS, "words": 1,
                      "voice": self.voice, "rate": self.rate, "pitch": self.pitch}
        cached = self.get_cached_result(input_data, cache_dir)
        if cached is not None:
            return cached

        audio_path = path_to_string(path) if path else self.get_audio_basename(input_data) + ".mp3"
        full = Path(cache_dir) / audio_path
        words = asyncio.run(_synthesize(input_text, self.voice, self.rate, self.pitch, full))
        cut = trim_silence(full)
        return {"input_text": text, "input_data": input_data, "original_audio": audio_path,
                "word_boundaries": to_word_boundaries(input_text, words, cut)}
