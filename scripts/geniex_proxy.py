#!/usr/bin/env python3
"""
Tiny reverse proxy in front of `geniex serve`.

Problem: GenieX's /v1/models advertises models WITH a precision suffix
(e.g. "qualcomm/Qwen3-VL-4B-Instruct:W4A16"), but /v1/chat/completions
only accepts the BARE model id (no suffix). Every OpenAI-compatible
frontend (Open WebUI, Live VLM WebUI, etc.) trusts /v1/models and sends
the tagged name back, which GenieX then rejects with:
    SDKError(Invalid input parameters or handle), quantization 'W4A16' not found

Fix: this proxy strips any ":PRECISION" suffix from the "model" field
on the way in, so frontends can keep using whatever they auto-discovered
and it just works.

Usage:
    python3 geniex_proxy.py
    # proxy listens on 18182, forwards (fixed) requests to real GenieX on 18181

Then point Open WebUI / Live VLM WebUI / any client at:
    http://127.0.0.1:18182/v1
instead of :18181/v1
"""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request

GENIEX_BACKEND = "http://127.0.0.1:18181"
PROXY_PORT = 18182

TAG_RE = re.compile(r":[A-Za-z0-9_]+$")  # strips trailing ":W4A16" etc.


class ProxyHandler(BaseHTTPRequestHandler):
    def _read_body(self):
        """Read the request body whether it's sent with Content-Length or
        chunked Transfer-Encoding (common for larger payloads like image frames)."""
        te = self.headers.get("Transfer-Encoding", "")
        if "chunked" in te.lower():
            chunks = []
            while True:
                size_line = self.rfile.readline().strip()
                if not size_line:
                    break
                chunk_size = int(size_line.split(b";")[0], 16)
                if chunk_size == 0:
                    self.rfile.readline()  # trailing CRLF after last chunk
                    break
                chunks.append(self.rfile.read(chunk_size))
                self.rfile.read(2)  # consume trailing CRLF after each chunk
            return b"".join(chunks)
        else:
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

    def _forward(self):
        body = self._read_body()

        # Fix the "model" field if this is a JSON request with one
        if body:
            try:
                data = json.loads(body)
                if isinstance(data, dict) and "model" in data and isinstance(data["model"], str):
                    fixed = TAG_RE.sub("", data["model"])
                    if fixed != data["model"]:
                        print(f"[proxy] rewriting model '{data['model']}' -> '{fixed}'")
                    data["model"] = fixed
                    body = json.dumps(data).encode()
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # not JSON, forward as-is (e.g. multipart image upload)

        url = GENIEX_BACKEND + self.path
        skip_headers = {"host", "content-length", "transfer-encoding"}
        out_headers = {k: v for k, v in self.headers.items() if k.lower() not in skip_headers}
        if body:
            out_headers["Content-Length"] = str(len(body))

        req = urllib.request.Request(
            url,
            data=body if body else None,
            method=self.command,
            headers=out_headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                # stream the response back (works for both plain and SSE streaming)
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"proxy error: {e}"}).encode())

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def log_message(self, format, *args):
        print(f"[proxy] {self.address_string()} - {format % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), ProxyHandler)
    print(f"GenieX fix-up proxy running on http://127.0.0.1:{PROXY_PORT}/v1")
    print(f"Forwarding to real GenieX server at {GENIEX_BACKEND}")
    server.serve_forever()
