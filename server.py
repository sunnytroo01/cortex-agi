"""
Cortex AGI — Web Server

Serves the chat UI and API endpoints.
Loads the latest checkpoint automatically.

Usage:
  python server.py                    # default (small config)
  python server.py --config large     # B200 config
  python server.py --checkpoint checkpoints/cortex_latest.pt
"""

import os
import json
import http.server
import socketserver
import torch
import time
import argparse

from cortex import Cortex, CortexConfig

# --- Parse args ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="small", choices=["small", "medium", "large", "xl"])
parser.add_argument("--checkpoint", default="checkpoints/cortex_latest.pt")
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

PORT = args.port
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("  CORTEX AGI")
print("=" * 50)

# Load model
if os.path.exists(args.checkpoint):
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    cortex = Cortex.load_checkpoint(args.checkpoint, device=DEVICE)
    print(f"  Loaded! Steps: {cortex.step_count:,}")
else:
    print(f"\n  No checkpoint found, creating fresh {args.config} model...")
    configs = {"small": CortexConfig.small, "medium": CortexConfig.medium,
               "large": CortexConfig.large, "xl": CortexConfig.xl}
    config = configs[args.config]()
    config.device = DEVICE
    cortex = Cortex(config).to(DEVICE)

stats = cortex.stats()
print(f"  Columns: {stats['columns']} | Regions: {stats['regions']}")
print(f"  Neurons: {stats['total_neurons']:,} | Synapses: {stats['total_synapses']:,}")
print(f"  Sparsity: {stats['active_pct']} | Device: {DEVICE}")
print(f"\n  http://localhost:{PORT}")
print("=" * 50)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(__file__) or ".", **kwargs)

    def do_POST(self):
        if self.path == "/api/chat":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                prompt = body.get("message", "")

                cortex.feed_text(prompt)
                response = cortex.generate(prompt, max_bytes=300, temperature=0.8)

                self._json_response(200, {
                    "response": response,
                    "stats": cortex.stats(),
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json_response(500, {"error": str(e)})

        elif self.path == "/api/feed":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                text = body.get("text", "")
                repeats = min(body.get("repeats", 5), 50)

                t0 = time.time()
                acc = 0
                for _ in range(repeats):
                    acc = cortex.feed_text(text)
                elapsed = time.time() - t0

                self._json_response(200, {
                    "accuracy": round(acc, 1),
                    "repeats": repeats,
                    "time": round(elapsed, 2),
                    "steps": cortex.step_count,
                    "stats": cortex.stats(),
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json_response(500, {"error": str(e)})
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/api/stats":
            self._json_response(200, cortex.stats())
        else:
            super().do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        if "/api/" in str(args[0]):
            super().log_message(format, *args)


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
