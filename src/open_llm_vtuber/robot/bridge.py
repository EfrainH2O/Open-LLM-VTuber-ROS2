"""Link between this VTuber and the teleoperation robot-server (CLAUDE.md 3.7, 3.8).

One VTuber serves every robot of the fleet. robot-server keeps one link here and tags every
message with the robot it is about ("robot": <id>):

- robot-server -> here: {"type":"robots","list":[{id,name,online}]}  the registered robots
                        {"type":"speech","robot","text"}   operator speech (committed Whisper text)
                        {"type":"heard","robot","text"}    someone near that robot spoke (its tablet
                                                           mic, Whisper on the server; autonomous only)
                        {"type":"teleop","robot","on"}     that robot is (no longer) teleoperated
                        {"type":"expression","robot","name"}  emotion from its operator's wheel
                        {"type":"spoke","robot","chars","ms"} a real Kachaka utterance, to learn its speed
                        {"type":"voice","robot","output"}  where that robot is heard: "robot" (its own
                                                           speaker, the default) or "tablet" (it has
                                                           no speech service)
                        {"type":"avatar","robot","name"}   its tablet's settings picked this avatar model
- here -> robot-server: {"type":"emotions","names"}        the default avatar's emotions (every robot)
                        {"type":"emotions","robot","names"}  that robot's avatar emotions: its tablet
                                                           picked its own model (avatar.py)
                        {"type":"avatar","robot","current","names"}  that robot's avatar model and the
                                                           ones it can pick (for its tablet's settings)
                        {"type":"speak","robot","text"}    that robot's Kachaka says this sentence now

Each robot tablet's WebView connects to /client-ws?robot_ticket=<ticket> and is that robot's
"robot client". robot-server hands the tablet the ticket (signed with VTUBER_TOKEN, see
verify_ticket), so the WebView never needs the tablet's own token. For a robot client, every
response (operator speech or LLM reply) is voiced by the robot's voice output (voice.py). By
default that is its Kachaka: audio payloads become muted mouth tracks timed to that Kachaka
(lipsync.py), and when the tablet starts playing a sentence (audio-play-start) the sentence goes
to that robot. A robot with no speech service is voiced by its tablet instead: the TTS audio is
played aloud there and nothing is sent to the robot. Its WebView has no microphone (the tablet app
owns it and sends it to Whisper on the server: "heard"), so upstream's start-mic greeting is not
sent to it; its VAD could only fail ("Failed to start VAD").

Robots never share context (hard rule): each robot has its own agent (the LLM's memory), kept
across WebView reconnects; its own chat-history folder (conf_uid "<conf_uid>@<robot id>"); and its
own heard queue, teleop flag, emotion, speech worker and Kachaka timing (RobotLink). Robot clients
cannot join groups, be invited into one, or switch character. A message for a missing or unknown
robot is dropped, never given to a default agent. Each robot also has its own avatar model, picked
on its tablet in the WebView's "Character Preset" menu, which lists models for a robot (avatar.py).

The AI moves its robot with the "kachaka" MCP tool (kachaka_mcp.py): robot-server POST
/vtuber/action -> that robot's tablet -> Kachaka gRPC. Each robot client's tool process is bound to
its robot here (_bind_robot_tool), so the LLM never chooses which robot it acts on.

Disabled (plain upstream behaviour) unless ROBOT_SERVER_WS and VTUBER_TOKEN are set.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from loguru import logger
from pydub import AudioSegment

from ..chat_history_manager import create_new_history, get_metadata
from ..live2d_model import Live2dModel
from ..config_manager.utils import scan_config_alts_directory
from .avatar import DEFAULT_FILE as AVATARS_FILE, PC_LABEL, AvatarStore, model_names, model_of_preset, preset_list, retarget_agent
from .external_speech import OPERATOR_SPEECH, ExternalSpeech
from .lipsync import KachakaTiming
# SAY_KEY, SOURCE_KEY, VOICE_KEY: extra display_text fields; the frontend echoes display_text in audio-play-start
from .voice import DEFAULT_OUTPUT, OUTPUTS, SAY_KEY, SOURCE_KEY, VOICE_KEY, VoiceOutput, output_of, say_text

if TYPE_CHECKING:
    from ..service_context import ServiceContext
    from ..websocket_handler import WebSocketHandler

# Conversation triggers that make the AI answer; refused on the robot while it is teleoperated.
AI_TRIGGERS = {"mic-audio-end", "text-input", "ai-speak-signal"}
# Would mix a robot's conversation with other clients' or load another character's memory.
GROUP_OPS = {"add-client-to-group", "remove-client-from-group"}
EMOTIONS_MAX = 8   # slices on the operator's emotion wheel
APPLY_AFTER_CONNECT_S = 4.0   # a reconnecting tablet needs its Live2D model loaded first
EXPRESSION_CLIP_MS = 100      # silent carrier for an expression change (see expression_payload)
ROBOT_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
LEGACY_ROBOT = "robot-1"      # what robot-server registers the old shared ROBOT_AUTH_TOKEN as
ROBOT_TOOL = "kachaka"        # the AI's robot tool in mcp_servers.json (kachaka_mcp.py)


def valid_robot_id(robot_id) -> bool:
    return isinstance(robot_id, str) and ROBOT_ID_RE.fullmatch(robot_id) is not None


def verify_ticket(secret: str, ticket: str, now: float | None = None) -> str | None:
    """The robot id of a valid, unexpired robot-server ticket, else None. Same format and check as
    robot-server's app/tokens.py: vt1.<robot id>.<expiry>.<base64url HMAC-SHA256(secret, "vt1.<id>.<expiry>")>"""
    parts = ticket.split(".")
    if not secret or len(parts) != 4 or parts[0] != "vt1" or not valid_robot_id(parts[1]):
        return None
    body = ".".join(parts[:3])
    sig = base64.urlsafe_b64encode(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    if not hmac.compare_digest(sig, parts[3]):
        return None
    try:
        expiry = int(parts[2])
    except ValueError:
        return None
    return parts[1] if expiry > (time.time() if now is None else now) else None


def robot_conf_uid(conf_uid: str, robot_id: str) -> str:
    """The robot's own chat-history folder: chat_history/<conf_uid>@<robot id>/."""
    return f"{conf_uid}@{robot_id}"


def http_origin(ws_url: str) -> str:
    """robot-server's HTTP origin from its WebSocket URL: ws://robot-server:8000/vtuber -> http://robot-server:8000."""
    u = urlsplit(ws_url)
    scheme = {"ws": "http", "wss": "https"}.get(u.scheme, u.scheme)
    return f"{scheme}://{u.netloc}" if u.netloc else ""


def _is_start_mic(data: str) -> bool:
    """Upstream's {"type": "control", "text": "start-mic"}, sent to every client on connect."""
    if len(data) > 64 or "start-mic" not in data:
        return False
    try:
        msg = json.loads(data)
    except ValueError:
        return False
    return isinstance(msg, dict) and msg.get("type") == "control" and msg.get("text") == "start-mic"


@dataclass
class RobotLink:
    """Everything the VTuber keeps for one robot. Nothing in here is shared with another robot."""
    robot_id: str
    timing: KachakaTiming          # this Kachaka's learned speaking rate
    voice: VoiceOutput = DEFAULT_OUTPUT   # where it is heard: its own speaker, or its tablet (voice.py)
    model: Live2dModel | None = None      # avatar model its tablet picked (avatar.py); None: the default
    uid: str | None = None         # its tablet WebView's client uid, while connected
    ctx: "ServiceContext | None" = None   # that client's context, while connected
    agent: object = None           # its own LLM agent (memory), kept across WebView reconnects
    history_uid: str = ""          # its current conversation, in its own history folder
    teleop: bool = False           # teleoperated: the AI may not act through this robot
    expression: str = ""           # emotion picked on its operator's wheel, kept on its avatar
    pending_heard: str = ""        # heard while it was answering: its next turn
    spoken_turn: asyncio.Task | None = None   # the turn the robot is speaking: hears nothing meanwhile
    speech: ExternalSpeech | None = None


class RobotWebSocket:
    """Wraps a robot client's WebSocket: audio payloads go through the robot's voice output
    (mouth tracks timed to its Kachaka, or the TTS audio as it is for the tablet's speaker)."""

    def __init__(self, ws, bridge: "RobotBridge", robot_id: str):
        self._ws = ws
        self._bridge = bridge
        self.robot_id = robot_id
        # Re-timing runs in a thread; the lock (FIFO) keeps messages in order, so e.g.
        # "backend-synth-complete" can never overtake the last sentence's audio.
        self._send_lock = asyncio.Lock()

    async def send_text(self, data: str) -> None:
        if _is_start_mic(data):
            return   # the WebView has no mic (see the module docstring)
        operator = OPERATOR_SPEECH.get()
        async with self._send_lock:
            if '"type": "audio"' in data[:40] or '"type":"audio"' in data[:40]:
                data = await asyncio.to_thread(self._bridge.transform_audio, data, operator, self.robot_id)
            await self._ws.send_text(data)
            # The frontend resets the avatar to its default expression when a turn ends (a task
            # queued by conversation-chain-end); queue the operator's emotion right after it.
            if '"conversation-chain-end"' in data and (emotion := self._bridge.expression_payload(self.robot_id)):
                await self._ws.send_text(emotion)

    def __getattr__(self, name):
        return getattr(self._ws, name)


class RobotBridge:
    def __init__(self, ws_handler: "WebSocketHandler"):
        self.ws_handler = ws_handler
        self.url = os.environ.get("ROBOT_SERVER_WS", "").strip()
        self.token = os.environ.get("VTUBER_TOKEN", "").strip()
        # legacy: tablets with the old app still send the shared ROBOT_AUTH_TOKEN (= robot-1)
        self.legacy_token = os.environ.get("ROBOT_AUTH_TOKEN", "").strip()
        self.enabled = bool(self.url and self.token)
        self.chars_per_sec = float(os.environ.get("KACHAKA_CHARS_PER_SEC", "6.0"))
        self.lead_ms = int(os.environ.get("KACHAKA_LEAD_MS", "400"))
        self.robots: dict[str, RobotLink] = {}
        self.by_uid: dict[str, str] = {}     # robot client uid -> robot id
        self.pending: dict[str, str] = {}    # client uid -> robot id, until its context is built
        self.known: set[str] | None = None   # robots registered on robot-server (None: not told yet)
        self.avatars = AvatarStore(os.environ.get("ROBOT_AVATARS_FILE", AVATARS_FILE).strip() or None)
        self._avatar_told: set[str] = set()   # robots told their avatar on this robot-server link
        self._conn = None
        self._task: asyncio.Task | None = None

    # --- robots -------------------------------------------------------------------------
    def link(self, robot_id: str) -> RobotLink:
        link = self.robots.get(robot_id)
        if link is None:
            link = self.robots[robot_id] = RobotLink(
                robot_id, KachakaTiming(chars_per_sec=self.chars_per_sec, lead_ms=self.lead_ms),
                model=self._saved_model(robot_id))
        return link

    def _saved_model(self, robot_id: str) -> Live2dModel | None:
        """The avatar model this robot's tablet picked before (avatar.py), if it is still listed."""
        name = self.avatars.get(robot_id)
        if not name:
            return None
        if name in model_names():
            try:
                return Live2dModel(name)
            except Exception as e:
                logger.warning(f"🎨 {robot_id}: avatar {name} not loaded ({e})")
        logger.warning(f"🎨 {robot_id}: saved avatar {name} is not in model_dict.json: default avatar")
        return None

    def avatar(self, link: RobotLink | None) -> Live2dModel | None:
        """The robot's own avatar model, else the default one (conf.yaml's live2d_model_name)."""
        if link is not None and link.model is not None:
            return link.model
        return getattr(getattr(self.ws_handler, "default_context_cache", None), "live2d_model", None)

    def client_link(self, client_uid: str) -> RobotLink | None:
        """The robot this client is the tablet of, None for any other client (e.g. a PC browser)."""
        robot_id = self.by_uid.get(client_uid)
        return self.robots.get(robot_id) if robot_id else None

    def _connection(self, link: RobotLink):
        return getattr(self.ws_handler, "client_connections", {}).get(link.uid) if link.uid else None

    # --- robot clients (the tablet WebViews) ---------------------------------------------
    def robot_id_of(self, websocket) -> str | None:
        """Which robot's tablet this /client-ws connection is, None for other clients."""
        if not self.enabled:
            return None
        params = websocket.query_params
        robot_id = verify_ticket(self.token, params.get("robot_ticket", ""))
        if robot_id is None:
            legacy = params.get("robot_token", "")
            if self.legacy_token and legacy and hmac.compare_digest(legacy.encode(), self.legacy_token.encode()):
                robot_id = LEGACY_ROBOT
        elif self.known is not None and robot_id not in self.known:
            logger.warning(f"🤖 Ticket for {robot_id}, which robot-server no longer lists: plain client")
            return None
        return robot_id

    def wrap(self, websocket, robot_id: str) -> RobotWebSocket:
        return RobotWebSocket(websocket, self, robot_id)

    def expect(self, client_uid: str, robot_id: str) -> None:
        """Called before the client's context is built, so isolate_context knows it is a robot."""
        self.pending[client_uid] = robot_id

    async def isolate_context(self, ctx: "ServiceContext", client_uid: str) -> None:
        """[robot fork hook] Give a robot client its robot's own memory: upstream hands every client
        the default context's agent, which would mix all robots' conversations."""
        robot_id = self.pending.get(client_uid)
        if robot_id is None:
            return
        link = self.link(robot_id)
        ctx.character_config.conf_uid = robot_conf_uid(ctx.character_config.conf_uid, robot_id)
        self._use_avatar(ctx, link.model)       # before the agent is built with its emotion tags
        self._bind_robot_tool(ctx, robot_id)   # before the agent is built with this context's tools
        if link.agent is None:
            ctx.agent_engine = None   # init_agent keeps an existing agent: build this robot's own
            await ctx.init_agent(ctx.character_config.agent_config, ctx.character_config.persona_prompt)
            link.agent = ctx.agent_engine
            logger.info(f"🤖 {robot_id}: own agent (memory) created")
        ctx.agent_engine = link.agent
        ctx.history_uid = link.history_uid
        if getattr(link.agent, "_live2d_model", ctx.live2d_model) is not ctx.live2d_model:
            await retarget_agent(ctx)   # the avatar was changed from the tablet while its page was closed

    @staticmethod
    def _use_avatar(ctx: "ServiceContext", model: Live2dModel | None) -> None:
        """This robot client's context shows `model` (None: keeps the default it was cloned with).
        Its conf_name becomes the model's name too: the WebView's "Character Preset" menu lists
        models for a robot and marks the entry named like conf_name (avatar.py)."""
        if model is not None:
            ctx.live2d_model = model
            ctx.character_config.live2d_model_name = model.live2d_model_name
        if ctx.live2d_model is not None:
            ctx.character_config.conf_name = ctx.live2d_model.live2d_model_name

    @staticmethod
    async def _send_model(websocket, client_uid: str, ctx: "ServiceContext") -> None:
        """Upstream's set-model-and-conf: the frontend (re)loads the context's model."""
        await websocket.send_text(json.dumps({
            "type": "set-model-and-conf", "model_info": ctx.live2d_model.model_info,
            "conf_name": ctx.character_config.conf_name, "conf_uid": ctx.character_config.conf_uid,
            "client_uid": client_uid,
        }))

    async def switch_avatar(self, websocket, client_uid: str, link: RobotLink, name: str) -> None:
        """[robot fork hook] This robot's avatar page picked another model in its menu."""
        error = await self.set_avatar(link, name, page=websocket)
        if error is not None:
            await websocket.send_text(json.dumps({"type": "error", "message": error}))
            if (ctx := self.ws_handler.client_contexts.get(client_uid)) is not None:
                await self._send_model(websocket, client_uid, ctx)   # the frontend dropped its model: same one back

    async def set_avatar(self, link: RobotLink, name: str, page=None) -> str | None:
        """This robot shows avatar model `name` from now on, asked from its avatar page's menu
        (`page`: that page's socket) or from its tablet's settings (robot-server "avatar"). Only this
        robot changes: its avatar, its agent's emotion tags and its operator's wheel. The character,
        its memory and its conversation stay, and the choice is saved for its next connections.
        An error text, or None."""
        if name not in model_names():
            error = f"unknown avatar model: {name}"
        else:
            try:
                model = Live2dModel(name)
                error = None
            except Exception as e:
                error = f"avatar model {name} could not be loaded: {e}"
        if error is not None:
            logger.warning(f"🎨 {link.robot_id}: {error}")
            await self.announce_avatar(link)   # its tablet shows the model it still has
            return error
        link.model = model
        self.avatars.set(link.robot_id, name)
        if link.expression and link.expression not in model.emo_map:
            link.expression = ""   # robot-server drops it too once it gets the new emotions
        ctx = link.ctx
        ws = page or self._connection(link)
        if ctx is not None:   # its avatar page is open: it shows the new model now
            self._use_avatar(ctx, model)
            await retarget_agent(ctx)
            if ws is not None:
                await self._send_model(ws, link.uid, ctx)
                await ws.send_text(json.dumps({"type": "config-switched", "message": f"Avatar: {name}"}))
        logger.info(f"🎨 {link.robot_id}: avatar model is now {name} (this robot only)")
        await self.announce_emotions(link)
        await self.announce_avatar(link)
        if link.expression and ws is not None:   # the operator's emotion, on the new model once it has loaded
            asyncio.get_running_loop().create_task(self.apply_expression(link.robot_id, delay=APPLY_AFTER_CONNECT_S))
        return None

    async def announce_avatar(self, link: RobotLink) -> None:
        """This robot's avatar and the ones it can pick, for its tablet's settings (via robot-server)."""
        if self._conn is None:
            return   # sent when the link is up again (_set_known)
        model = self.avatar(link)
        msg = {"type": "avatar", "robot": link.robot_id, "names": model_names(),
               "current": model.live2d_model_name if model is not None else None}
        try:
            await self._conn.send(json.dumps(msg, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"{link.robot_id}'s avatar list not sent: {e}")

    def _bind_robot_tool(self, ctx: "ServiceContext", robot_id: str) -> None:
        """This robot client's "kachaka" tool acts on this robot only: its process (started lazily by
        the context's own MCP client, from the context's own server registry) gets the robot id and
        the way to robot-server. Other clients' processes get neither and answer "no robot"."""
        registry = getattr(ctx, "mcp_server_registery", None)
        server = registry.get_server(ROBOT_TOOL) if registry is not None else None
        if server is None:
            return   # MCP off, or the tool is not configured
        server.env = {**(server.env or {}), "KACHAKA_ROBOT_ID": robot_id,
                      "ROBOT_SERVER_URL": http_origin(self.url), "VTUBER_TOKEN": self.token}

    def attach(self, client_uid: str, robot_id: str) -> None:
        self.pending.pop(client_uid, None)
        link = self.link(robot_id)
        if link.speech:
            link.speech.cancel()
        if link.uid and link.uid != client_uid:
            self.by_uid.pop(link.uid, None)   # the robot's previous WebView: newest wins
        link.uid = client_uid
        link.ctx = self.ws_handler.client_contexts.get(client_uid)
        self.by_uid[client_uid] = robot_id
        link.speech = ExternalSpeech(self.ws_handler, client_uid)
        logger.info(f"🤖 {robot_id}: tablet connected as VTuber client {client_uid}")
        if link.expression:
            asyncio.get_running_loop().create_task(self.apply_expression(robot_id, delay=APPLY_AFTER_CONNECT_S))

    def detach(self, client_uid: str) -> None:
        self.pending.pop(client_uid, None)
        robot_id = self.by_uid.pop(client_uid, None)
        link = self.robots.get(robot_id) if robot_id else None
        if link is None or link.uid != client_uid:
            return
        if link.speech:
            link.speech.cancel()
        if link.ctx is not None:   # keep its conversation for the next connection of this robot
            link.history_uid = link.ctx.history_uid or ""
        link.uid, link.ctx, link.speech = None, None, None
        logger.info(f"🤖 {robot_id}: tablet disconnected")

    async def intercept(self, websocket, client_uid: str, data: dict) -> bool:
        """[robot fork hook] Messages that would break a robot's isolation, and its avatar menu
        (a model list instead of the characters, avatar.py). True = handled here."""
        kind = data.get("type")
        link = self.client_link(client_uid)
        if kind in GROUP_OPS:
            target = data.get("invitee_uid") or data.get("target_uid")
            if link is not None or target in self.by_uid:
                logger.info(f"🤖 Refused {kind}: robot conversations are never shared")
                await websocket.send_text(json.dumps({"type": "error", "message": "robot clients cannot be grouped"}))
                return True
            return False
        if link is None:
            return await self._pc_menu(websocket, client_uid, kind, data)
        if kind == "fetch-configs":
            # its "Character Preset" menu lists avatar models: the character is fixed, the model is its own
            await websocket.send_text(json.dumps({"type": "config-files", "configs": preset_list(model_names())}))
            return True
        if kind == "switch-config" and (name := model_of_preset(data.get("file"))) is not None:
            await self.switch_avatar(websocket, client_uid, link, name)
            return True
        if kind == "switch-config":
            logger.info(f"🤖 {link.robot_id}: switch-config refused (would load another memory)")
            await websocket.send_text(json.dumps({"type": "error", "message": "the robot's character is fixed"}))
            return True
        if kind == "create-new-history":
            # The frontend asks for a new history on every connect; a reconnecting robot keeps its
            # conversation (its agent still remembers it). Only the first one creates its history.
            ctx = self.ws_handler.client_contexts.get(client_uid)
            if ctx is None:
                return False
            conf_uid = ctx.character_config.conf_uid
            if not link.history_uid or not get_metadata(conf_uid, link.history_uid):
                link.history_uid = create_new_history(conf_uid)
            ctx.history_uid = link.history_uid
            await websocket.send_text(json.dumps({"type": "new-history-created", "history_uid": link.history_uid}))
            return True
        return False

    async def _pc_menu(self, websocket, client_uid: str, kind, data: dict) -> bool:
        """Any other client (a PC browser): its menu lists the characters as upstream, then the
        avatar models ("Avatar: <name>"). A model picked there changes only this page, saved nowhere."""
        ctx = self.ws_handler.client_contexts.get(client_uid)
        if ctx is None:
            return False
        if kind == "fetch-configs":
            configs = scan_config_alts_directory(ctx.system_config.config_alts_dir) + preset_list(model_names(), PC_LABEL)
            await websocket.send_text(json.dumps({"type": "config-files", "configs": configs}))
            return True
        if kind != "switch-config" or (name := model_of_preset(data.get("file"))) is None:
            return False   # a character: upstream's switch
        try:
            if name not in model_names():
                raise KeyError(name)
            model = Live2dModel(name)
        except Exception as e:
            await websocket.send_text(json.dumps({"type": "error", "message": f"avatar model {name}: {e}"}))
            await self._send_model(websocket, client_uid, ctx)
            return True
        ctx.live2d_model = model
        ctx.character_config.live2d_model_name = name
        ctx.character_config.conf_name = PC_LABEL + name   # marks it in the menu
        await self._send_model(websocket, client_uid, ctx)
        await websocket.send_text(json.dumps({"type": "config-switched", "message": f"Avatar: {name}"}))
        logger.info(f"🎨 client {client_uid}: avatar model {name} (this page only)")
        return True

    def transform_audio(self, data: str, operator: bool = False, robot_id: str = "") -> str:
        """Runs in a worker thread: one audio payload, prepared for that robot's voice output."""
        try:
            link = self.link(robot_id)
            payload = json.loads(data)
            display = payload.get("display_text") or {}
            say = say_text(display.get("text", ""))
            if display:
                display[SOURCE_KEY] = "operator" if operator else "ai"
                display[VOICE_KEY] = link.voice.name
            return json.dumps(link.voice.prepare(payload, say, link.timing))
        except Exception as e:  # never lose a sentence because of the mouth track
            logger.warning(f"Could not prepare audio for {robot_id}'s voice: {e}")
            return data

    async def on_audio_play_start(self, client_uid: str, data: dict) -> None:
        """A robot's tablet just started playing a sentence: the robot says it now. With the robot's
        own speaker that is a "speak" command; with the tablet's speaker the audio is the voice."""
        link = self.client_link(client_uid)
        if link is None:
            return
        display = data.get("display_text") or {}
        voice = output_of(display)   # the output this payload was prepared for
        say = voice.said(display)
        if link.teleop and display.get(SOURCE_KEY) != "operator":
            # no speak command; a tablet that is the voice mutes the sentence itself
            logger.info(f"{link.robot_id} teleoperated: AI sentence not spoken: {say}")
            return
        if say:
            logger.info(f"🗣️ {link.robot_id} says ({voice.name}): {say}")
            self._robot_speaks(link)
            if voice.command:
                await self.send_speak(link.robot_id, say)

    # --- teleoperation gate ---------------------------------------------------------------
    def set_teleop(self, robot_id: str, on: bool) -> None:
        link = self.link(robot_id)
        if on == link.teleop:
            return
        link.teleop = on
        logger.info(f"🎮 {robot_id}: teleoperation {'ON: AI blocked on the robot' if on else 'OFF: AI allowed'}")
        if on and link.uid:
            # cut an AI answer that is still being generated (operator speech keeps going)
            task = self.ws_handler.current_conversation_tasks.get(link.uid)
            if task and not task.done() and (link.speech is None or task is not link.speech.task):
                task.cancel()

    async def reject_if_blocked(self, websocket, client_uid: str, msg_type: str) -> bool:
        """[robot fork hook] Refuse AI conversation triggers from a robot while it is teleoperated."""
        link = self.client_link(client_uid)
        if not (link is not None and link.teleop and msg_type in AI_TRIGGERS):
            return False
        logger.info(f"{link.robot_id} teleoperated: ignored '{msg_type}' from its tablet")
        self.ws_handler.received_data_buffers[client_uid] = self.ws_handler.received_data_buffers[client_uid][:0]
        # let the frontend leave its "thinking" state
        await websocket.send_text(json.dumps({"type": "control", "text": "conversation-chain-end"}))
        return True

    # --- robot hearing ----------------------------------------------------------------------
    async def on_heard(self, robot_id: str, text: str) -> None:
        """[robot fork hook] Someone near the robot said `text` (its tablet mic, Whisper on the server).
        It becomes that robot client's next turn, as if typed on the tablet, so the AI answers through
        its avatar and Kachaka. Heard while an answer is being prepared: it waits, merged with anything
        else heard meanwhile, for when that turn ends. Once the robot starts speaking the answer, that
        queue is dropped and nothing heard counts until the turn ends (_robot_speaks). The place
        for another answer policy later (wake word, LLM decides), and for attaching the tablet
        camera's latest still ("images")."""
        link = self.link(robot_id)
        ws = self._connection(link)
        if link.teleop or ws is None:
            logger.info(f"👂 {robot_id} heard but not answered ({'teleoperated' if link.teleop else 'no tablet'}): {text}")
            return
        running = self.ws_handler.current_conversation_tasks.get(link.uid)
        if running is not None and not running.done() and running is link.spoken_turn:
            logger.info(f"👂 {robot_id} heard while the robot speaks, dropped: {text}")
            return
        if running is not None and not running.done():
            if not link.pending_heard:
                running.add_done_callback(lambda _: asyncio.ensure_future(self._heard_turn(robot_id)))
            link.pending_heard = f"{link.pending_heard} {text}".strip()
            logger.info(f"👂 {robot_id} heard while answering, next turn: {text}")
            return
        logger.info(f"👂 {robot_id} heard: {text}")
        await ws.send_text(json.dumps({"type": "user-input-transcription", "text": text}))
        await self.ws_handler._handle_conversation_trigger(ws, link.uid, {"type": "text-input", "text": text})

    def _robot_speaks(self, link: RobotLink) -> None:
        """The robot speaks a sentence of the running turn (its Kachaka, or its tablet's speaker):
        what was heard and not yet answered is dropped, and so is anything heard until the turn ends
        (the tablet also drops the speech it had not sent yet, see robot-server media.py)."""
        running = self.ws_handler.current_conversation_tasks.get(link.uid) if link.uid else None
        if running is None or running.done():
            return
        link.spoken_turn = running
        if link.pending_heard:
            logger.info(f"👂 {link.robot_id} speaks, dropped what it had not answered: {link.pending_heard}")
            link.pending_heard = ""

    async def _heard_turn(self, robot_id: str) -> None:
        link = self.link(robot_id)
        text, link.pending_heard = link.pending_heard, ""
        if text:
            await self.on_heard(robot_id, text)   # checks again: teleop, tablet, another turn started

    # --- robot-server link ----------------------------------------------------------------
    async def start(self) -> None:
        if self._task is not None:   # FastAPI may run a router's startup handler more than once
            return
        if not self.enabled:
            logger.info("Robot bridge disabled (set ROBOT_SERVER_WS and VTUBER_TOKEN to enable)")
            return
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        from websockets.asyncio.client import connect

        backoff = 1.0
        while True:
            try:
                async with connect(self.url, additional_headers={"Authorization": f"Bearer {self.token}"}) as conn:
                    self._conn, backoff = conn, 1.0
                    self._avatar_told = set()   # this robot-server learns every robot's avatar again
                    logger.info(f"🤖 Connected to robot-server {self.url}")
                    await conn.send(json.dumps({"type": "emotions", "names": self.emotions()}))
                    for robot_id in list(self.avatars.models):   # robots that picked their own avatar
                        await self.announce_emotions(self.link(robot_id))
                    async for raw in conn:
                        await self._on_message(raw)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Robot-server link down ({e}); retrying in {backoff:.0f}s")
            self._conn = None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    def _emo_map(self, link: RobotLink | None = None) -> dict:
        """Live2D emotionMap (name -> expression index): the robot client's model, else the robot's
        own avatar, else the default."""
        ctx = link.ctx if link is not None else None
        for model in (getattr(ctx, "live2d_model", None), self.avatar(link)):
            emo_map = getattr(model, "emo_map", None)
            if emo_map:
                return emo_map
        return {}

    def emotions(self, link: RobotLink | None = None) -> list[str]:
        """The avatar's emotion names (Live2D emotionMap keys) for the operator's wheel, at most 8:
        this robot's avatar, else the default one (every robot that did not pick its own)."""
        return list(self._emo_map(link))[:EMOTIONS_MAX]

    async def announce_emotions(self, link: RobotLink) -> None:
        """This robot's avatar emotions, for its operator's wheel (robot-server keeps them per robot)."""
        if self._conn is None:
            return   # sent when the link is up again (_run)
        try:
            await self._conn.send(json.dumps({"type": "emotions", "robot": link.robot_id, "names": self.emotions(link)}))
        except Exception as e:
            logger.warning(f"{link.robot_id}'s emotions not sent: {e}")

    def expression_payload(self, robot_id: str) -> str | None:
        """The robot operator's emotion as a frontend message. The frontend only sets expressions
        from an audio payload's actions, so this is a short silent clip with no display_text (no
        subtitle, no audio-play-start, so Kachaka says nothing)."""
        link = self.robots.get(robot_id)
        index = self._emo_map(link).get(link.expression) if link is not None and link.expression else None
        if index is None:
            return None
        buf = io.BytesIO()
        AudioSegment.silent(duration=EXPRESSION_CLIP_MS, frame_rate=16000).export(buf, format="wav")
        return json.dumps({
            "type": "audio", "audio": base64.b64encode(buf.getvalue()).decode(),
            "volumes": [0.0] * (EXPRESSION_CLIP_MS // 20), "slice_length": 20,
            "display_text": None, "actions": {"expressions": [index]}, "forwarded": False,
            SAY_KEY: "",   # marks it muted for the tablet's WebView hook, like the mouth tracks
        })

    async def apply_expression(self, robot_id: str, delay: float = 0.0) -> None:
        """Show the operator's emotion on this robot's avatar now (after `delay` seconds)."""
        if delay:
            await asyncio.sleep(delay)
        link = self.robots.get(robot_id)
        ws = self._connection(link) if link is not None else None
        if ws is None or (data := self.expression_payload(robot_id)) is None:
            return
        try:
            await ws.send_text(data)
        except Exception as e:  # tablet gone; reapplied when it reconnects
            logger.debug(f"expression not sent: {e}")

    async def _on_message(self, raw) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        if not isinstance(msg, dict):
            return
        kind = msg.get("type")
        if kind == "robots" and isinstance(msg.get("list"), list):
            await self._set_known({r.get("id") for r in msg["list"] if isinstance(r, dict)})
            return
        robot_id = msg.get("robot")
        if not valid_robot_id(robot_id):
            logger.debug(f"robot-server message without a robot dropped: {kind}")
            return   # never handed to a default agent
        if kind == "speech" and isinstance(msg.get("text"), str):
            text = msg["text"].strip()
            if not text:
                return
            link = self.link(robot_id)
            if link.speech:
                link.speech.say(text)                  # through its avatar, the robot speaks on play start
            else:
                # No avatar page on that tablet, so no TTS audio can play there: the robot says it
                # itself if it can, whatever its voice output is (a robot with no speech refuses it).
                await self.send_speak(robot_id, text)
        elif kind == "voice":
            # [robot fork hook] where this robot is heard: chosen on its tablet (the "Kachaka voice"
            # switch, off by default) and only "robot" if the robot can speak
            voice = OUTPUTS.get(msg.get("output"))
            link = self.link(robot_id)
            if voice is not None and voice is not link.voice:
                link.voice = voice
                logger.info(f"🔈 {robot_id} voice output: {voice.name}")
        elif kind == "heard" and isinstance(msg.get("text"), str) and msg["text"].strip():
            await self.on_heard(robot_id, msg["text"].strip())
        elif kind == "teleop":
            self.set_teleop(robot_id, bool(msg.get("on")))
        elif kind == "expression" and isinstance(msg.get("name"), str):
            # [robot fork hook] operator emotion: shown on that robot's avatar and kept after each turn
            self.link(robot_id).expression = msg["name"]
            logger.info(f"🎭 {robot_id} operator emotion: {msg['name']}")
            await self.apply_expression(robot_id)
        elif kind == "avatar" and isinstance(msg.get("name"), str):
            # [robot fork hook] its tablet's settings picked another avatar model (avatar.py)
            await self.set_avatar(self.link(robot_id), msg["name"])
        elif kind == "spoke":
            timing = self.link(robot_id).timing
            timing.observe(int(msg.get("chars", 0)), int(msg.get("ms", 0)))
            logger.debug(f"{robot_id} Kachaka speaking rate now {timing.chars_per_sec:.2f} chars/s")

    async def _set_known(self, ids: set) -> None:
        """The robots robot-server lists now. A tablet of a robot that was removed is cut off. Each
        newly listed robot is told its avatar and the ones it can pick (its tablet's settings)."""
        self.known = {i for i in ids if valid_robot_id(i)}
        for robot_id in sorted(self.known - self._avatar_told):   # once per robot-server link
            self._avatar_told.add(robot_id)
            await self.announce_avatar(self.link(robot_id))
        for link in list(self.robots.values()):
            if link.robot_id not in self.known and (ws := self._connection(link)) is not None:
                logger.warning(f"🤖 {link.robot_id} is no longer registered: closing its tablet's VTuber client")
                with contextlib.suppress(Exception):
                    await ws.close(code=4401)

    async def send_speak(self, robot_id: str, text: str) -> None:
        if self._conn is None:
            logger.warning(f"Robot-server not connected: {robot_id}'s Kachaka cannot speak")
            return
        try:
            await self._conn.send(json.dumps({"type": "speak", "robot": robot_id, "text": text}, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"speak not sent: {e}")
