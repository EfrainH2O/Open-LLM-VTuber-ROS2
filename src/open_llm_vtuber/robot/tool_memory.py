"""Tool calls stay in the agent's memory.

Upstream's OpenAI tool loop sends the tool call and its result to the LLM within the turn, but
keeps only the spoken text in the agent's memory. Later turns then see replies such as "moving to
the dining room... please wait... arrived" with no tool call behind them, and the model imitates
them: it narrates moving the robot instead of calling the tool, and the robot never moves.
remember_tool_round keeps the whole round instead: the assistant message with its tool calls, then
each result, as the OpenAI API expects them.

Only in-process memory: chat history files keep their human/AI text (upstream format), so after a
VTuber restart the tool rounds of older turns are gone again.

tools_for_round caps the tool rounds of one turn: upstream's loop asks the LLM again after every
result, with no limit, and a model that answers each result with another call never speaks.
"""

from __future__ import annotations

from loguru import logger

# Tool rounds in one turn before the LLM must answer in words. A request needs at most two
# (robot_status for the names, then the action); a third leaves room for one more.
MAX_TOOL_ROUNDS = 3


def tools_for_round(tools: list | None, rounds_done: int) -> list | None:
    """The tools offered to the next LLM request of a turn: none once the turn has had
    MAX_TOOL_ROUNDS tool rounds, so that request is answered in words. nemotron-3.5-lightning, for
    example, sometimes answers the result of call_operator with call_operator again or with
    stop_robot, instead of telling the person that staff were called."""
    if rounds_done < MAX_TOOL_ROUNDS:
        return tools
    logger.info(f"🔧 {rounds_done} tool rounds in this turn: the LLM answers without tools now")
    return None


def remember_tool_round(agent, call: dict, results: list[dict]) -> None:
    """Add one tool round to `agent`'s memory. A call is kept only with its result (a call without
    one is rejected by the API on the next turn); with no result at all, only the text, as upstream."""
    answered = {r.get("tool_call_id") for r in results if r.get("role") == "tool"}
    calls = [c for c in call.get("tool_calls") or [] if c.get("id") in answered]
    text = call.get("content") or ""
    if not calls:
        if text:
            agent._add_message(text, "assistant")
        return
    ids = {c["id"] for c in calls}
    # content "" (not None): upstream's interrupt handling reads the last assistant message as text
    agent._memory.append({"role": "assistant", "content": text, "tool_calls": calls})
    agent._memory.extend(r for r in results if r.get("role") == "tool" and r.get("tool_call_id") in ids)
