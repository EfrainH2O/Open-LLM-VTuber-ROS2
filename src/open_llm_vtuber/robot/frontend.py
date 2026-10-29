"""The VTuber page as the robot fork serves it: only the avatar at start, centered on its face.

Upstream's frontend (the `frontend` submodule, a built bundle) opens with the chat-history sidebar
and the text-input footer expanded. Their state is two `useState` defaults in `AppContent`, saved
nowhere, so `collapse_panels` flips them in the bundle as it is served: both panels start
collapsed, and their chevrons still open them. `face_center.js` is appended to the bundle: with
`faceCenter` in the model's model_dict.json entry, each loaded model is moved once so its face is
centered (see that file). The submodule itself stays untouched.

The patched bundle is served from memory with its own ETag, and index.html asks for it under a
new URL (`?robot=<hash>`), so a browser or tablet WebView that cached upstream's bundle cannot keep
showing the open panels. If a frontend update changes the panels' code, they start open and a
warning is logged; if it is no longer the bundle this expects (no `AppContent`), both files are
served as they are (tests/robot_fork_test.py checks the bundle in the submodule).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from loguru import logger
from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.staticfiles import NotModifiedResponse

# function AppContent(){const[showSidebar,..]=useState(true),[isFooterCollapsed,..]=useState(false)
_PANELS = re.compile(
    r"(function AppContent\(\)\{const\[[\w$]+,[\w$]+\]=(?:[\w$]+\.)?useState\()!0"
    r"(\),\[[\w$]+,[\w$]+\]=(?:[\w$]+\.)?useState\()!1\)"
)
_BUNDLE_SRC = re.compile(r'(<script type="module"[^>]*\bsrc=")([^"?]+\.js)(")')
_APP = "function AppContent()"   # marks the app bundle
FACE_CENTER = Path(__file__).with_name("face_center.js").read_text(encoding="utf-8")


def collapse_panels(js: str) -> str | None:
    """The bundle with the sidebar closed and the footer collapsed at start, None if not found."""
    out, n = _PANELS.subn(r"\g<1>!1\g<2>!0)", js, count=1)
    return out if n else None


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class RobotFrontend:
    """Mixin for the frontend's StaticFiles mount (server.py): index.html and the app bundle it
    loads are served patched (panels collapsed, face centered); every other file as upstream does."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # bundle path -> ((mtime, size), patched body, its sha1); body None: not the app bundle
        self._bundles: dict[str, tuple[tuple[int, int], bytes | None, str]] = {}
        self._warned = False

    def file_response(self, full_path, stat_result, scope, status_code: int = 200) -> Response:
        patched = self._patch(os.path.realpath(full_path)) if status_code == 200 else None
        if patched is None:
            return super().file_response(full_path, stat_result, scope, status_code)
        body, sha, media_type = patched
        headers = Headers(headers={"etag": f'"{sha}"', "cache-control": "no-cache"})
        if self.is_not_modified(headers, Headers(scope=scope)):
            return NotModifiedResponse(headers)
        return Response(body, media_type=media_type, headers=dict(headers))

    def _patch(self, path: str) -> tuple[bytes, str, str] | None:
        root = os.path.realpath(self.directory)
        if path == os.path.join(root, "index.html"):
            html = _read(path)
            src = _BUNDLE_SRC.search(html)
            bundle = src and self._bundle(os.path.join(root, os.path.normpath(src.group(2))))
            if not bundle:
                return None
            html = _BUNDLE_SRC.sub(rf"\g<1>\g<2>?robot={bundle[1][:12]}\g<3>", html, count=1).encode()
            return html, hashlib.sha1(html).hexdigest(), "text/html"
        if path.endswith(".js") and path.startswith(os.path.join(root, "assets") + os.sep):
            bundle = self._bundle(path)
            return (*bundle, "application/javascript") if bundle else None
        return None

    def _bundle(self, path: str) -> tuple[bytes, str] | None:
        """The patched app bundle and its sha1, cached until the file changes; None if this file
        is not the app bundle."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        cached = self._bundles.get(path)
        if cached is None or cached[0] != key:
            js = _read(path)
            body = None
            if _APP in js:
                collapsed = collapse_panels(js)
                if collapsed is None and not self._warned:
                    self._warned = True
                    logger.warning("robot fork: the frontend bundle changed, its panels start open (robot/frontend.py)")
                body = ((collapsed or js) + "\n" + FACE_CENTER).encode()
            cached = (key, body, hashlib.sha1(body).hexdigest() if body else "")
            self._bundles[path] = cached
        return (cached[1], cached[2]) if cached[1] else None
