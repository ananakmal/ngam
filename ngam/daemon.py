"""
ngam.daemon
===========
Lightweight, ultra-fast in-memory HTTP daemon for the ngam Universal
Decision Engine. Eliminates process startup and model loading latency,
delivering sub-5ms decision evaluation via localhost loopback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from ngam.engine import UniversalDecider

logger = logging.getLogger("ngam.daemon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global singleton decider instance
_GLOBAL_DECIDER: Optional[UniversalDecider] = None


def get_global_decider(preferred_provider: Optional[str] = None) -> UniversalDecider:
    global _GLOBAL_DECIDER
    if _GLOBAL_DECIDER is None:
        logger.info("Initializing in-memory singleton ngam UniversalDecider...")
        _GLOBAL_DECIDER = UniversalDecider(preferred_provider=preferred_provider)
        # Pre-warm the ONNX computation graph
        try:
            _GLOBAL_DECIDER.decide("warmup prompt", choice=["a", "b"])
            logger.info(f"ngam UniversalDecider pre-warmed successfully (Provider: {_GLOBAL_DECIDER.active_provider})")
        except Exception as e:
            logger.warning(f"Decider pre-warm warning: {e}")
    return _GLOBAL_DECIDER


class DecisionHTTPRequestHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP handler for ngam decision requests."""

    # Use HTTP/1.1 with keep-alive for sub-millisecond connection reuse
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Suppress routine access logging in production for minimal overhead
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Full CORS preflight handler responding with 204 No Content."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def do_GET(self):
        if self.path == "/healthz":
            decider = get_global_decider()
            self._send_json_response(200, {
                "status": "ok",
                "provider": decider.active_provider,
                "model": getattr(decider, "model_name", "laya-onnx"),
                "warm": True,
            })
        else:
            self._send_json_response(404, {"error": f"Endpoint not found: {self.path}"})

    def do_POST(self):
        if self.path == "/v1/decide":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json_response(400, {"error": "Missing or empty request body"})
                return

            try:
                body_bytes = self.rfile.read(content_length)
                payload = json.loads(body_bytes.decode("utf-8"))
            except Exception as parse_err:
                self._send_json_response(400, {"error": f"Invalid JSON payload: {parse_err}"})
                return

            prompt = payload.get("prompt")
            if not prompt:
                self._send_json_response(400, {"error": "Field 'prompt' is required"})
                return

            choice = payload.get("choice")
            score = payload.get("score")
            noul = payload.get("noul")
            question = payload.get("question")

            if not (choice or score or noul):
                self._send_json_response(400, {"error": "Must specify at least one of: choice, score, noul"})
                return

            try:
                decider = get_global_decider()
                result = decider.decide(
                    prompt=prompt,
                    choice=choice,
                    score=score,
                    noul=noul,
                    question=question,
                )
                self._send_json_response(200, result.model_dump())
            except Exception as run_err:
                logger.error(f"Inference error: {run_err}", exc_info=True)
                self._send_json_response(500, {"error": f"Inference execution failed: {run_err}"})

        elif self.path == "/v1/shutdown":
            self._send_json_response(200, {"status": "shutting_down"})
            # Schedule graceful shutdown in separate thread
            def _delayed_shutdown():
                time.sleep(0.1)
                self.server.shutdown()
            threading.Thread(target=_delayed_shutdown, daemon=True).start()
        else:
            self._send_json_response(404, {"error": f"Endpoint not found: {self.path}"})


def create_server(
    host: str = "127.0.0.1",
    port: int = 8045,
    decider: Optional[UniversalDecider] = None,
) -> ThreadingHTTPServer:
    """Create and pre-warm a ThreadingHTTPServer instance."""
    global _GLOBAL_DECIDER
    if decider is not None:
        _GLOBAL_DECIDER = decider
    else:
        get_global_decider()

    server = ThreadingHTTPServer((host, port), DecisionHTTPRequestHandler)
    return server


def run_daemon(host: str = "127.0.0.1", port: int = 8045, provider: Optional[str] = None) -> None:
    """Start the daemon server listening until interrupted."""
    global _GLOBAL_DECIDER
    _GLOBAL_DECIDER = get_global_decider(preferred_provider=provider)
    server = create_server(host=host, port=port, decider=_GLOBAL_DECIDER)
    logger.info(f"⚡ ngam Decision Daemon listening on http://{host}:{port} (Active: {_GLOBAL_DECIDER.active_provider})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received, closing daemon...")
    finally:
        server.server_close()
        logger.info("Daemon server terminated.")


def main(args: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="ngam In-Memory Ultra-Fast Decision Daemon")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8045, help="Port (default: 8045)")
    parser.add_argument("-p", "--provider", type=str, default=None, help="Preferred provider (dml, coreml, cuda, cpu)")
    parsed = parser.parse_args(args)

    run_daemon(host=parsed.host, port=parsed.port, provider=parsed.provider)
    return 0


if __name__ == "__main__":
    sys.exit(main())
