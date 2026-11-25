"""The AI's robot tool: an MCP server (stdio) named "kachaka" in mcp_servers.json (CLAUDE.md 3.7).

Each robot tool call becomes one robot action: POST robot-server /vtuber/action -> that robot's
tablet -> Kachaka's gRPC API. The server cannot reach Kachaka itself (only the tablet is on the
robot's LAN). `call_operator` instead asks robot-server (POST /vtuber/call-operator) to call a human,
like the tablet's call button, with the reason the LLM writes (CLAUDE.md 3.9).

The process acts on one robot only. The LLM never names the robot: the bridge (bridge.py,
`_bind_robot_tool`) starts this server for a robot client with

    KACHAKA_ROBOT_ID   that client's robot id
    ROBOT_SERVER_URL   robot-server's HTTP origin, from ROBOT_SERVER_WS
    VTUBER_TOKEN       the shared secret robot-server checks

Without them (any other client, or listing the tools) every tool answers that no robot is connected.
robot-server refuses all but robot_status while an operator teleoperates the robot, and so does the tablet.

Runs as a script (stdlib + the mcp SDK only), so it starts the same in Docker and in a source checkout:
    python src/open_llm_vtuber/robot/kachaka_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP

HTTP_TIMEOUT_S = 20   # robot-server gives up on the tablet after 12 s
REASON_MAX_CHARS = 300   # what robot-server keeps of a call's reason

mcp = FastMCP(
    "kachaka",
    instructions="Moves the Kachaka robot you are on, and calls a human operator to it. Actions start "
    "the robot and return at once; check robot_status for names and for what the robot is doing.",
)


def _post(path: str, body: dict) -> str:
    """POST robot-server `path` for the bound robot; its {"ok", "text"} answer as one line for the LLM."""
    robot = os.environ.get("KACHAKA_ROBOT_ID", "")
    base = os.environ.get("ROBOT_SERVER_URL", "").rstrip("/")
    if not robot or not base:
        return "Not available: this conversation is not connected to a robot."
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps({"robot": robot, **body}).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('VTUBER_TOKEN', '')}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as res:
            out = json.load(res)
    except urllib.error.HTTPError as e:
        return f"Not done: the robot server refused the request (HTTP {e.code})."
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return f"Not done: the robot server is not reachable ({e})."
    text = str(out.get("text") or "")
    return f"OK: {text}" if out.get("ok") is True else f"Not done: {text}"


async def _act(action: str, **args) -> str:
    """One robot action, run by the robot's tablet. Empty arguments are left out."""
    args = {k: v for k, v in args.items() if v}
    return await asyncio.to_thread(_post, "/vtuber/action", {"action": action, "args": args})


@mcp.tool()
async def robot_status() -> str:
    """The robot's battery, what it is doing now or last did, and the exact names of its saved
    locations and shelves. Call it before moving if you do not know the names."""
    return await _act("status")


@mcp.tool()
async def move_to_location(location: str) -> str:
    """Drive the robot to one of its saved locations, given by its exact name from robot_status.
    Returns as soon as the robot starts moving."""
    return await _act("move_to", location=location)


@mcp.tool()
async def visit_locations(locations: list[str], wait_seconds: float = 0) -> str:
    """Drive the robot through several saved locations in the given order (exact names from
    robot_status, at most 10), staying wait_seconds (0-300) at each one before the last, where it
    stays. Returns as soon as the robot starts toward the first; robot_status shows how far it got."""
    return await _act("route", locations=locations, wait_s=wait_seconds)


@mcp.tool()
async def bring_shelf(shelf: str, location: str = "") -> str:
    """Fetch a shelf: the robot drives to it, docks under it and carries it to a saved location
    (empty: the shelf's own home), staying docked. Names as in robot_status."""
    return await _act("bring_shelf", shelf=shelf, location=location)


@mcp.tool()
async def return_shelf() -> str:
    """Take the shelf the robot is carrying back to its home and leave it there."""
    return await _act("return_shelf")


@mcp.tool()
async def go_charge() -> str:
    """Send the robot back to its charger (a carried shelf is returned home first)."""
    return await _act("charge")


@mcp.tool()
async def stop_robot() -> str:
    """Cancel what the robot is doing now (moving, a route, carrying a shelf, going to charge)."""
    return await _act("cancel")


@mcp.tool()
async def call_operator(reason: str) -> str:
    """Call a human member of staff, who can take over this robot and talk through it. Use it when
    someone asks for a person or staff, when you cannot help them (a question you cannot answer, a
    complaint, something the robot cannot do), or when someone seems hurt, unwell, lost or in danger.
    reason: one short sentence for the staff saying who needs what. Afterwards tell the person that
    staff have been called. Calling again while a call waits does not make a new one."""
    return await asyncio.to_thread(_post, "/vtuber/call-operator", {"reason": reason.strip()[:REASON_MAX_CHARS]})


if __name__ == "__main__":
    mcp.run()
