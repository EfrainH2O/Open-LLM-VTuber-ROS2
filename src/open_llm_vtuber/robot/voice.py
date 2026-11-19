"""Where a robot's sentences are heard: the voice outputs of the response pipeline.

    inputs (who writes the text)         pipeline (upstream, shared)                outputs (who is heard)
    operator speech (ExternalSpeech) -+  sentence split, emotion tags, subtitle,  +- KachakaVoice "robot"
    the AI's answer (the agent)      -+> TTS engine (tts_config), Live2D payload -+- TabletVoice  "tablet"

Every sentence, whatever wrote it, leaves the pipeline as one audio payload for the robot's tablet.
The robot's output then decides two things: what that payload becomes (`prepare`), and whether the
sentence also goes to the robot as a "speak" command when the tablet starts playing it (`command`).

Which output a robot uses comes from robot-server ({"type":"voice","robot","output"}), which has it
from the robot's tablet. There it is a settings switch, "Kachaka voice": off by default (the tablet
is the voice), and turned on only after the robot passed a voice test. A robot with no speech
service (a ROS 2 robot behind the translator with `tts.backend: none`) is always voiced by its
tablet. Each robot has its own (RobotLink.voice). Until a tablet has reported, and for tablet apps
from before the switch, the output is the robot.

A new output (another speaker, a ROS audio topic...) is a new subclass listed in OUTPUTS. A new
input only has to send its text through the pipeline, like ExternalSpeech does. A new TTS engine
(e.g. a local one in its own container) is upstream's tts_config: nothing here depends on it.
"""

from __future__ import annotations

from loguru import logger

from ..utils.tts_preprocessor import tts_filter as filter_text
from .lipsync import KachakaTiming, retime_payload, speakable_chars, synth_payload

SAY_KEY = "robot_say"       # display_text field: what Kachaka says; its presence mutes the payload on the tablet
SOURCE_KEY = "robot_source"   # display_text field: "operator" or "ai"
VOICE_KEY = "robot_voice"   # display_text field: the output the payload was prepared for


def say_text(display: str) -> str:
    """What is spoken for a displayed sentence: same filtering as the TTS input
    (drops [emotion] tags, brackets, asterisks...)."""
    return filter_text(
        text=display, remove_special_char=True, ignore_brackets=True,
        ignore_parentheses=True, ignore_asterisks=True, ignore_angle_brackets=True,
    ).strip()


class VoiceOutput:
    """One place a robot's sentences can be heard. Stateless: one instance serves every robot."""

    name = ""
    command = False   # the sentence is also sent to the robot ("speak") when the tablet starts playing it

    def prepare(self, payload: dict, say: str, timing: KachakaTiming) -> dict:
        """The audio payload as this output needs it. Runs in a worker thread."""
        return payload

    def said(self, display: dict) -> str:
        """The spoken text of a payload the tablet started playing (its echoed display_text)."""
        return say_text(display.get("text") or "")


class KachakaVoice(VoiceOutput):
    """The robot's own speaker (Kachaka's SpeakCommand, or the translator's TTS). The TTS audio is
    only a muted mouth track on the tablet, re-timed to the robot's speaking rate (lipsync.py)."""

    name = "robot"
    command = True

    def prepare(self, payload: dict, say: str, timing: KachakaTiming) -> dict:
        if display := payload.get("display_text"):
            display[SAY_KEY] = say
        if not say:
            return payload
        if payload.get("audio"):
            return retime_payload(payload, timing.estimate_ms(say), timing.lead_ms)
        # TTS failed (e.g. edge-tts refused): the robot still speaks, the mouth is synthesized
        return synth_payload(payload, speakable_chars(say), timing.estimate_ms(say), timing.lead_ms)

    def said(self, display: dict) -> str:
        return (display.get(SAY_KEY) or "").strip()


class TabletVoice(VoiceOutput):
    """The tablet's speaker (what the tablets choose by default): the TTS audio is the voice. The payload
    goes out as the TTS made it (no SAY_KEY, so the tablet plays it aloud), and the avatar lip-syncs
    to it as upstream does. The tablet itself pauses the robot's hearing while it plays."""

    name = "tablet"

    def prepare(self, payload: dict, say: str, timing: KachakaTiming) -> dict:
        if say and not payload.get("audio"):
            logger.warning(f"TTS gave no audio and the tablet is this robot's only voice: shown, not heard: {say}")
        return payload


OUTPUTS: dict[str, VoiceOutput] = {v.name: v for v in (KachakaVoice(), TabletVoice())}
DEFAULT_OUTPUT = OUTPUTS[KachakaVoice.name]


def output_of(display: dict) -> VoiceOutput:
    """The output a played payload was prepared for (payloads from before VOICE_KEY: the robot)."""
    return OUTPUTS.get(display.get(VOICE_KEY), DEFAULT_OUTPUT)
