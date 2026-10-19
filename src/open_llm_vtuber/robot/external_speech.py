"""External speech: text from outside (the teleoperator) spoken BY the avatar.

It reuses the normal response pipeline (sentence split -> emotion tags -> display text ->
TTS -> Live2D payloads) and only replaces the LLM as the source of the text, so everything a
response does (subtitles, expressions, lip sync, history) also happens for operator speech.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import TYPE_CHECKING

from loguru import logger

from ..agent.output_types import AudioOutput, SentenceOutput
from ..agent.transformers import (
    actions_extractor,
    display_processor,
    sentence_divider,
    tts_filter,
)
from ..chat_history_manager import store_message
from ..conversations.conversation_utils import (
    cleanup_conversation,
    finalize_conversation_turn,
    process_agent_output,
    send_conversation_start_signals,
)
from ..conversations.tts_manager import TTSTaskManager

if TYPE_CHECKING:
    from ..service_context import ServiceContext
    from ..websocket_handler import WebSocketHandler

# Operator segments arrive about once a second while talking; keep them in one conversation
# turn (one chain start/end on the frontend) until nothing new came for this long.
IDLE_S = 1.5

# True inside the operator-speech worker (and the TTS sender tasks it creates, which inherit it):
# lets the bridge tell operator sentences from AI sentences when teleoperation blocks the AI.
OPERATOR_SPEECH: ContextVar[bool] = ContextVar("operator_speech", default=False)


def _pipeline(ctx: "ServiceContext"):
    """Same decorator stack as BasicMemoryAgent._chat_function_factory, fed by a fixed text."""
    settings = {}
    try:
        settings = ctx.character_config.agent_config.agent_settings.basic_memory_agent.model_dump()
    except AttributeError:
        pass

    @tts_filter(ctx.character_config.tts_preprocessor_config)
    @display_processor()
    @actions_extractor(ctx.live2d_model)
    @sentence_divider(
        faster_first_response=settings.get("faster_first_response", True),
        segment_method=settings.get("segment_method", "pysbd"),
        valid_tags=["think"],
    )
    async def passthrough(text: str):
        yield text

    return passthrough


class ExternalSpeech:
    def __init__(self, ws_handler: "WebSocketHandler", client_uid: str):
        self.ws_handler = ws_handler
        self.client_uid = client_uid
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.task: asyncio.Task | None = None

    def say(self, text: str) -> None:
        self.queue.put_nowait(text)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())

    def cancel(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()

    async def _run(self) -> None:
        OPERATOR_SPEECH.set(True)
        # Keep taking turns while text is waiting. Text that arrives while a turn waits for the
        # tablet to finish playing is picked up here; say() does not start a second worker while
        # this one runs. No await between the empty() check and returning, so nothing slips through.
        while await self._turn():
            if self.queue.empty():
                return

    async def _turn(self) -> bool:
        """One conversation turn on the tablet. False if the robot client is gone."""
        uid = self.client_uid
        ctx = self.ws_handler.client_contexts.get(uid)
        ws = self.ws_handler.client_connections.get(uid)
        if ctx is None or ws is None:
            while not self.queue.empty():   # nobody to show it to: drop instead of spinning
                self.queue.get_nowait()
            return False
        # The operator has priority over whatever the LLM was saying on this client.
        running = self.ws_handler.current_conversation_tasks.get(uid)
        if running and running is not self.task and not running.done():
            running.cancel()
        self.ws_handler.current_conversation_tasks[uid] = self.task

        send = ws.send_text
        tts_manager = TTSTaskManager()
        spoken = ""
        try:
            await send_conversation_start_signals(send)
            while True:
                try:
                    text = await asyncio.wait_for(self.queue.get(), timeout=IDLE_S)
                except asyncio.TimeoutError:
                    break
                logger.info(f"🎙️ Operator speech: {text}")
                async for out in _pipeline(ctx)(text):
                    if isinstance(out, (SentenceOutput, AudioOutput)):
                        spoken += await process_agent_output(
                            output=out,
                            character_config=ctx.character_config,
                            live2d_model=ctx.live2d_model,
                            tts_engine=ctx.tts_engine,
                            websocket_send=send,
                            tts_manager=tts_manager,
                            translate_engine=ctx.translate_engine,
                        ) or ""
            await finalize_conversation_turn(tts_manager=tts_manager, websocket_send=send, client_uid=uid)
            self._remember(ctx, spoken)
            return True
        except asyncio.CancelledError:
            logger.info("Operator speech interrupted")
            raise
        finally:
            cleanup_conversation(tts_manager, "🎙️")

    @staticmethod
    def _remember(ctx: "ServiceContext", spoken: str) -> None:
        """What the avatar said goes into history and the LLM's memory, as if it had answered."""
        if not spoken:
            return
        if ctx.history_uid:
            store_message(
                conf_uid=ctx.character_config.conf_uid,
                history_uid=ctx.history_uid,
                role="ai",
                content=spoken,
                name=ctx.character_config.character_name,
                avatar=ctx.character_config.avatar,
            )
        add = getattr(ctx.agent_engine, "_add_message", None)
        if callable(add):
            add(spoken, "assistant")
