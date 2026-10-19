"""Mouth timing for a voice that is not ours: Kachaka speaks, the avatar only mouths along.

Kachaka's API takes text and speaks it on the robot; it returns no audio. So for the robot
tablet the VTuber still synthesizes each sentence with its own TTS, but only as a mouth
envelope: the WAV is re-timed to Kachaka's expected duration, prefixed with Kachaka's start
latency, and played muted on the tablet (Live2D lip-syncs from the WAV data, not from the
speaker). Pitch does not matter because nobody hears it, so re-timing is a plain sample-rate
change, which keeps the syllable rhythm of real speech.
"""

from __future__ import annotations

import base64
import io

from pydub import AudioSegment

from ..utils.stream_audio import _get_volume_by_chunks


def speakable_chars(text: str) -> int:
    """Letters/digits only (kana and kanji count): a proxy for how long speech takes."""
    return sum(1 for ch in text if ch.isalnum())


class KachakaTiming:
    """Kachaka's speaking rate, learned from real utterance durations reported by the tablet."""

    def __init__(self, chars_per_sec: float = 6.0, lead_ms: int = 400):
        self.chars_per_sec = chars_per_sec
        self.lead_ms = lead_ms

    def estimate_ms(self, text: str) -> int:
        """Lead-in + speaking time, as the mouth track length."""
        n = speakable_chars(text)
        return self.lead_ms + int(1000 * n / self.chars_per_sec) if n else 0

    def observe(self, chars: int, ms: int) -> None:
        """Moving average from a measured utterance (StartCommand -> done on the robot)."""
        speak_ms = ms - self.lead_ms
        if chars < 4 or speak_ms < 300:
            return
        rate = min(20.0, max(2.0, chars / (speak_ms / 1000)))
        self.chars_per_sec = 0.8 * self.chars_per_sec + 0.2 * rate


def synth_mouth_track(chars: int, target_ms: int, lead_ms: int, rate: int = 16000) -> AudioSegment:
    """A speech-like envelope (one burst per character) for when TTS gave no audio: lead-in
    silence, then `chars` syllables spread over the rest. Muted on the tablet, so only its
    loudness shape matters to the lip sync."""
    import numpy as np

    body_s = max(200, target_ms - lead_ms) / 1000
    t = np.arange(int(body_s * rate)) / rate
    syll = t * max(1, chars) / body_s                       # syllable index over time
    depth = np.random.default_rng(chars).uniform(0.5, 1.0, max(1, chars) + 1)
    env = np.sin(np.pi * (syll % 1.0)) ** 2 * depth[syll.astype(int)]
    wave = (0.5 * env * np.sin(2 * np.pi * 180 * t) * 32767).astype(np.int16)
    body = AudioSegment(wave.tobytes(), frame_rate=rate, sample_width=2, channels=1)
    return AudioSegment.silent(duration=lead_ms, frame_rate=rate) + body


def synth_payload(payload: dict, chars: int, target_ms: int, lead_ms: int) -> dict:
    """Fill a silent payload (TTS failed) with a synthetic mouth track, so the tablet still
    shows the subtitle, moves the mouth and reports audio-play-start (Kachaka speaks)."""
    if target_ms <= 0:
        return payload
    out = synth_mouth_track(chars, target_ms, lead_ms)
    buf = io.BytesIO()
    out.export(buf, format="wav")
    slice_ms = payload.get("slice_length") or 20
    return {**payload, "audio": base64.b64encode(buf.getvalue()).decode(),
            "volumes": _get_volume_by_chunks(out, slice_ms), "slice_length": slice_ms}


def retime_payload(payload: dict, target_ms: int, lead_ms: int) -> dict:
    """Stretch the payload's WAV to `target_ms` (lead-in silence + speech) and recompute volumes."""
    if not payload.get("audio") or target_ms <= 0:
        return payload
    audio = AudioSegment.from_file(io.BytesIO(base64.b64decode(payload["audio"])), format="wav")
    if len(audio) == 0:
        return payload
    body_ms = max(200, target_ms - lead_ms)
    rate = int(audio.frame_rate * len(audio) / body_ms)
    rate = min(192000, max(4000, rate))
    stretched = audio._spawn(audio.raw_data, overrides={"frame_rate": rate})
    lead = AudioSegment.silent(duration=lead_ms, frame_rate=rate)
    lead = lead.set_channels(audio.channels).set_sample_width(audio.sample_width)
    out = lead + stretched
    buf = io.BytesIO()
    out.export(buf, format="wav")
    slice_ms = payload.get("slice_length") or 20
    try:
        volumes = _get_volume_by_chunks(out, slice_ms)
    except ValueError:
        volumes = []
    return {**payload, "audio": base64.b64encode(buf.getvalue()).decode(), "volumes": volumes}
