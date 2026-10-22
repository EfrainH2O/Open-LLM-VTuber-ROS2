"""The VTuber page starts showing only the avatar.

Upstream's frontend (the `frontend` submodule, a built bundle) opens with the chat-history sidebar
and the text-input footer expanded. Their state is two `useState` defaults in `AppContent`, saved
nowhere, so `collapse_panels` flips them in the bundle as it is served: both panels start
collapsed, and their chevrons still open them. The submodule itself stays untouched.

The patched bundle is served from memory with its own ETag, and index.html asks for it under a
new URL (`?robot=<hash>`), so a browser or tablet WebView that cached upstream's bundle cannot keep
showing the open panels. If a frontend update changes that code, both files are served as they
are and a warning is logged (tests/robot_fork_test.py checks the bundle in the submodule).
"""

from __future__ import annotations

import hashlib
import os
import re

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


def collapse_panels(js: str) -> str | None:
    """The bundle with the sidebar closed and the footer collapsed at start, None if not found."""
    out, n = _PANELS.subn(r"\g<1>!1\g<2>!0)", js, count=1)
    return out if n else None


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class CollapsedPanels:
    """Mixin for the frontend's StaticFiles mount (server.py): index.html and the app bundle it
    loads are served patched; every other file as upstream does."""

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
        is not the app bundle (or upstream changed the code and it cannot be patched)."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        cached = self._bundles.get(path)
        if cached is None or cached[0] != key:
            js = _read(path)
            patched = collapse_panels(js)
            if patched is None and "function AppContent()" in js and not self._warned:
                self._warned = True
                logger.warning("robot fork: the frontend bundle changed, its panels start open (robot/frontend.py)")
            body = patched.encode() if patched is not None else None
            cached = (key, body, hashlib.sha1(body).hexdigest() if body else "")
            self._bundles[path] = cached
        return (cached[1], cached[2]) if cached[1] else None
