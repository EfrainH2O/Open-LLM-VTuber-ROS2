"""Each robot's own avatar model (CLAUDE.md 3.7, 3.8): one robot's tablet changes its Live2D
model without touching any other robot's.

The tablet's WebView shows upstream's settings menu "Character Preset" (General tab). For a robot
client the bridge fills it with the avatar models of model_dict.json instead of the characters
(`fetch-configs` -> `config-files`, one entry "model:<name>" per model), and picking one
(`switch-config` with that file) changes only that robot: the avatar on its tablet, the emotion
tags its agent is told to use and takes out of its replies, and its operator's emotion wheel.
The character, the robot's memory and its conversation stay; other robots and other clients are
not touched. The same choice is in the tablet app's settings (tab "Avatar"): the tablet sends it
through robot-server ({"type":"avatar","robot","name"}), and gets each robot's avatar and the list
back ({"type":"avatar","robot","current","names"}).

A PC browser's menu lists the characters as upstream, then the models ("Avatar: <name>"). Picking
a model there changes only that page's avatar, saved nowhere; its agent (the shared one) keeps the
default model's emotion tags, so a model should use the same emotionMap names (ours all do).

The choice is kept per robot in a small JSON file (ROBOT_AVATARS_FILE, default
robot_state/avatars.json; a Docker volume in our compose), so it survives tablet reloads and
VTuber restarts. A robot that never chose, or whose model left model_dict.json, shows conf.yaml's
live2d_model_name.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..service_context import ServiceContext

PRESET_PREFIX = "model:"   # marks a menu entry as an avatar model, not a character file
DEFAULT_FILE = "robot_state/avatars.json"


def model_names(model_dict_path: str = "model_dict.json") -> list[str]:
    """The avatar models a robot can pick: every model_dict.json entry with a name and a url."""
    try:
        with open(model_dict_path, encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"🎨 No avatar models listed ({model_dict_path}: {e})")
        return []
    if not isinstance(entries, list):
        return []
    return [e["name"] for e in entries
            if isinstance(e, dict) and isinstance(e.get("name"), str) and e["name"] and e.get("url")]


def preset_list(names: list[str], label: str = "") -> list[dict]:
    """"Character Preset" menu entries (`config-files`) for avatar models: a robot client's whole
    menu (no label), or added after the characters in a PC browser's (label PC_LABEL)."""
    return [{"filename": PRESET_PREFIX + name, "name": label + name} for name in names]


PC_LABEL = "Avatar: "   # a PC browser's menu also lists characters: tell the models apart


def model_of_preset(filename) -> str | None:
    """The model a `switch-config` file names ("model:<name>"), None for a character file."""
    if isinstance(filename, str) and filename.startswith(PRESET_PREFIX):
        return filename[len(PRESET_PREFIX):] or None
    return None


class AvatarStore:
    """robot id -> the name of the avatar model its tablet picked. `path` None: in memory only."""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.models: dict[str, str] = {}
        if self.path is not None and self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.models = {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
            except (OSError, ValueError, AttributeError) as e:
                logger.warning(f"🎨 Saved avatar choices not read ({self.path}: {e}): every robot starts with the default")

    def get(self, robot_id: str) -> str | None:
        return self.models.get(robot_id)

    def set(self, robot_id: str, name: str) -> None:
        self.models[robot_id] = name
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.models, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)   # never a half-written file
        except OSError as e:
            logger.warning(f"🎨 {robot_id}'s avatar choice not saved ({e}): it lasts until the VTuber restarts")


async def retarget_agent(ctx: "ServiceContext") -> bool:
    """The context's agent now works with ctx.live2d_model: the emotion tags listed in its system
    prompt and the expressions taken out of its replies. Its memory is kept (BasicMemoryAgent; any
    other agent keeps the old model's tags, and the avatar still changes)."""
    agent = ctx.agent_engine
    if not all(hasattr(agent, a) for a in ("_live2d_model", "_chat_function_factory", "set_system")):
        logger.warning(f"🎨 {type(agent).__name__} cannot change its emotion tags: they stay the old model's")
        return False
    system = await ctx.construct_system_prompt(ctx.character_config.persona_prompt)
    agent._live2d_model = ctx.live2d_model
    agent.set_system(system)
    agent.chat = agent._chat_function_factory()   # its actions_extractor holds the model
    ctx.system_prompt = system
    return True
