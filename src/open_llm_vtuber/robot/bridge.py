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
- here -> robot-server: {"type":"emotions","names"}        the avatar's emotions (same character everywhere)
                        {"type":"speak","robot","text"}    that robot's Kachaka says this sentence now

Each robot tablet's WebView connects to /client-ws?robot_ticket=<ticket> and is that robot's
"robot client". robot-server hands the tablet the ticket (signed with VTUBER_TOKEN, see
verify_ticket), so the WebView never needs the tablet's own token. For a robot client, every
response (operator speech or LLM reply) is voiced by its Kachaka: audio payloads become muted
mouth tracks timed to that Kachaka (lipsync.py), and when the tablet starts playing a sentence
(audio-play-start) the sentence goes to that robot. Its WebView has no microphone (the tablet app
owns it and sends it to Whisper on the server: "heard"), so upstream's start-mic greeting is not
sent to it; its VAD could only fail ("Failed to start VAD").

Robots never share context (hard rule): each robot has its own agent (the LLM's memory), kept
across WebView reconnects; its own chat-history folder (conf_uid "<conf_uid>@<robot id>"); and its
own heard queue, teleop flag, emotion, speech worker and Kachaka timing (RobotLink). Robot clients
cannot join groups, be invited into one, or switch character. A message for a missing or unknown
robot is dropped, never given to a default agent.

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
from ..utils.tts_preprocessor import tts_filter as filter_text
from .external_speech import OPERATOR_SPEECH, ExternalSpeech
from .lipsync import KachakaTiming, retime_payload, speakable_chars, synth_payload

if TYPE_CHECKING:
    from ..service_context import ServiceContext
    from ..websocket_handler import WebSocketHandler

SAY_KEY = "robot_say"   # extra display_text field: the frontend echoes display_text in audio-play-start
SOURCE_KEY = "robot_source"   # "operator" or "ai"
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


def _say_text(display: str) -> str:
    """What Kachaka should say for a displayed sentence: same filtering as the TTS input
    (drops [emotion] tags, brackets, asterisks...)."""
    return filter_text(
        text=display, remove_special_char=True, ignore_brackets=True,
        ignore_parentheses=True, ignore_asterisks=True, ignore_angle_brackets=True,
    ).strip()


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
    uid: str | None = None         # its tablet WebView's client uid, while connected
    ctx: "ServiceContext | None" = None   # that client's context, while connected
    agent: object = None           # its own LLM agent (memory), kept across WebView reconnects
    history_uid: str = ""          # its current conversation, in its own history folder
    teleop: bool = False           # teleoperated: the AI may not act through this robot
    expression: str = ""           # emotion picked on its operator's wheel, kept on its avatar
    pending_heard: str = ""        # heard while it was answering: its next turn
    speech: ExternalSpeech | None = None


class RobotWebSocket:
    """Wraps a robot client's WebSocket: audio payloads become mouth tracks timed to its Kachaka."""

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
        self._conn = None
        self._task: asyncio.Task | None = None

    # --- robots -------------------------------------------------------------------------
    def link(self, robot_id: str) -> RobotLink:
        link = self.robots.get(robot_id)
        if link is None:
            link = self.robots[robot_id] = RobotLink(
                robot_id, KachakaTiming(chars_per_sec=self.chars_per_sec, lead_ms=self.lead_ms))
        return link

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
        self._bind_robot_tool(ctx, robot_id)   # before the agent is built with this context's tools
        if link.agent is None:
            ctx.agent_engine = None   # init_agent keeps an existing agent: build this robot's own
            await ctx.init_agent(ctx.character_config.agent_config, ctx.character_config.persona_prompt)
            link.agent = ctx.agent_engine
            logger.info(f"🤖 {robot_id}: own agent (memory) created")
        ctx.agent_engine = link.agent
        ctx.history_uid = link.history_uid

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
        """[robot fork hook] Messages that would break a robot's isolation. True = handled here."""
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
            return False
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

    def transform_audio(self, data: str, operator: bool = False, robot_id: str = "") -> str:
        """Runs in a worker thread: re-time one audio payload for that robot's Kachaka."""
        try:
            timing = self.link(robot_id).timing
            payload = json.loads(data)
            display = payload.get("display_text") or {}
            say = _say_text(display.get("text", ""))
            if display:
                display[SAY_KEY] = say
                display[SOURCE_KEY] = "operator" if operator else "ai"
            if payload.get("audio") and say:
                payload = retime_payload(payload, timing.estimate_ms(say), timing.lead_ms)
            elif say:   # TTS failed (e.g. edge-tts refused): Kachaka still speaks, mouth synthesized
                payload = synth_payload(payload, speakable_chars(say), timing.estimate_ms(say), timing.lead_ms)
            return json.dumps(payload)
        except Exception as e:  # never lose a sentence because of the mouth track
            logger.warning(f"Could not re-time audio for Kachaka: {e}")
            return data

    async def on_audio_play_start(self, client_uid: str, data: dict) -> None:
        """A robot's tablet just started mouthing a sentence: its Kachaka says it now."""
        link = self.client_link(client_uid)
        if link is None:
            return
        display = data.get("display_text") or {}
        say = (display.get(SAY_KEY) or "").strip()
        if link.teleop and display.get(SOURCE_KEY) != "operator":
            logger.info(f"{link.robot_id} teleoperated: AI sentence not spoken on Kachaka: {say}")
            return
        if say:
            logger.info(f"🗣️ {link.robot_id} Kachaka says: {say}")
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
        its avatar and Kachaka. Heard while an answer still runs: it waits, merged with anything else
        heard meanwhile, and goes in when that turn ends. The place for another answer policy later
        (wake word, LLM decides), and for attaching the tablet camera's latest still ("images")."""
        link = self.link(robot_id)
        ws = self._connection(link)
        if link.teleop or ws is None:
            logger.info(f"👂 {robot_id} heard but not answered ({'teleoperated' if link.teleop else 'no tablet'}): {text}")
            return
        running = self.ws_handler.current_conversation_tasks.get(link.uid)
        if running is not None and not running.done():
            if not link.pending_heard:
                running.add_done_callback(lambda _: asyncio.ensure_future(self._heard_turn(robot_id)))
            link.pending_heard = f"{link.pending_heard} {text}".strip()
            logger.info(f"👂 {robot_id} heard while answering, next turn: {text}")
            return
        logger.info(f"👂 {robot_id} heard: {text}")
        await ws.send_text(json.dumps({"type": "user-input-transcription", "text": text}))
        await self.ws_handler._handle_conversation_trigger(ws, link.uid, {"type": "text-input", "text": text})

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
                    logger.info(f"🤖 Connected to robot-server {self.url}")
                    await conn.send(json.dumps({"type": "emotions", "names": self.emotions()}))
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
        """Live2D emotionMap (name -> expression index): the robot client's model, else the default."""
        ctx = link.ctx if link is not None else None
        for source in (ctx, getattr(self.ws_handler, "default_context_cache", None)):
            emo_map = getattr(getattr(source, "live2d_model", None), "emo_map", None)
            if emo_map:
                return emo_map
        return {}

    def emotions(self) -> list[str]:
        """The avatar's emotion names (Live2D emotionMap keys) for the operators' wheel, at most 8.
        Every robot shows the same character, so one list serves all of them."""
        return list(self._emo_map())[:EMOTIONS_MAX]

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
                link.speech.say(text)                  # through its avatar, Kachaka speaks on play start
            else:
                await self.send_speak(robot_id, text)  # no avatar on that tablet: speak directly
        elif kind == "heard" and isinstance(msg.get("text"), str) and msg["text"].strip():
            await self.on_heard(robot_id, msg["text"].strip())
        elif kind == "teleop":
            self.set_teleop(robot_id, bool(msg.get("on")))
        elif kind == "expression" and isinstance(msg.get("name"), str):
            # [robot fork hook] operator emotion: shown on that robot's avatar and kept after each turn
            self.link(robot_id).expression = msg["name"]
            logger.info(f"🎭 {robot_id} operator emotion: {msg['name']}")
            await self.apply_expression(robot_id)
        elif kind == "spoke":
            timing = self.link(robot_id).timing
            timing.observe(int(msg.get("chars", 0)), int(msg.get("ms", 0)))
            logger.debug(f"{robot_id} Kachaka speaking rate now {timing.chars_per_sec:.2f} chars/s")

    async def _set_known(self, ids: set) -> None:
        """The robots robot-server lists now. A tablet of a robot that was removed is cut off."""
        self.known = {i for i in ids if valid_robot_id(i)}
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
