#!/usr/bin/env python3
"""Static server for the PathSnap evaluator UI + local OSM tile proxy.

Why this exists: on the campus network the browser does not trust the Sophos
TLS-inspection certificate, so public CDNs (unpkg, jsdelivr) and HTTPS map
tiles fail with ERR_CERT_AUTHORITY_INVALID.  Leaflet is therefore vendored in
`vendor/leaflet`, and this server fetches map tiles server-side (where we can
use an unverified context) and serves them to the browser over plain HTTP.

Run:
    python3 ui/serve.py                # http://0.0.0.0:8501
    python3 ui/serve.py --port 8502
"""
import argparse
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
TILE_RE = re.compile(r"^/tiles/(\d{1,2})/(\d+)/(\d+)\.png$")
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_CACHE = Path(
    os.environ.get("PATHSNAP_TILE_CACHE", Path(tempfile.gettempdir()) / "pathsnap_tiles")
)
# The campus gateway MITMs TLS with a private CA; verification is intentional
# only for the map-tile upstream, which is public data.
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE
USER_AGENT = "PathSnap-evaluator/1.0 (local demo; OSM tiles)"


def fetch_tile(z, x, y):
    """Return PNG bytes for a tile, using the on-disk cache when possible."""
    if z > 19 or x >= 2**z or y >= 2**z:
        return None
    cached = TILE_CACHE / str(z) / str(x) / f"{y}.png"
    if cached.exists():
        return cached.read_bytes()
    url = TILE_URL.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as resp:
        data = resp.read()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self):
        m = TILE_RE.match(self.path)
        if m:
            z, x, y = (int(v) for v in m.groups())
            try:
                data = fetch_tile(z, x, y)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self.send_error(502, f"tile upstream failed: {exc}")
                return
            if data is None:
                self.send_error(404, "tile out of range")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def do_HEAD(self):
        m = TILE_RE.match(self.path)
        if m:
            z, x, y = (int(v) for v in m.groups())
            try:
                data = fetch_tile(z, x, y)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self.send_error(502, f"tile upstream failed: {exc}")
                return
            self.send_response(200 if data else 404)
            if data:
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            return
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        super().do_HEAD()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PathSnap UI on http://{args.host}:{args.port}  (serving {UI_DIR})")
    print(f"Tile proxy cache: {TILE_CACHE}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
