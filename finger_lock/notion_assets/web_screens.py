"""Screenshots of the safe's web page in every state, using the real web.html and a stand-in
server (the board isn't needed). The camera picture is one of my real training pictures."""
import glob
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# camera view: red square on the desk with a real training picture of a hand inside it
hand = cv2.imread(sorted(glob.glob(str(APP / "laptop" / "data" / "3" / "*.jpg")))[-40])
frame = np.full((480, 640, 3), (60, 40, 215), np.uint8)
frame[380:, :] = (150, 175, 200)
frame[99:99 + 255, 34:34 + 537] = cv2.resize(hand, (537, 255))
JPEG = cv2.imencode(".jpg", cv2.resize(frame, (320, 240)))[1].tobytes()

base = {"mode": "locked", "entered": 0, "last": None, "confidences": [], "tries_left": 3,
        "lockout_seconds": 0, "lockout_length": 10, "tester": "", "reading": None}
reading = {"of": 7.0, "sure": 0.65}
SCENES = {
    "1_locked": {},
    "2_enter_show": {"mode": "listening", "reading": {**reading, "conf": 0, "votes": 0, "lead": 0, "phase": "show"}},
    "3_enter_reading": {"mode": "listening", "entered": 1, "confidences": [0.82],
                        "reading": {**reading, "conf": 0.71, "votes": 3.2, "lead": 3, "phase": "reading"}},
    "4_enter_away": {"mode": "listening", "entered": 2, "confidences": [0.82, 0.74],
                     "reading": {**reading, "conf": 0, "votes": 0, "lead": 0, "phase": "away"}},
    "5_wrong": {"last": "wrong", "tries_left": 2, "confidences": [0.82, 0.74, 0.69]},
    "6_lockout": {"last": "lockout", "tries_left": 0, "lockout_seconds": 7},
    "7_open": {"mode": "unlocked", "last": "open", "confidences": [0.82, 0.74, 0.77], "tester": "Sam"},
}
current = {"scene": "1_locked"}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            body, kind = JPEG, "image/jpeg"
        elif self.path.startswith("/state"):
            body, kind = json.dumps({**base, **SCENES[current["scene"]]}).encode(), "application/json"
        else:
            body, kind = (APP / "python" / "web.html").read_bytes(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
for scene in SCENES:
    current["scene"] = scene
    out = Path(f"web_{scene}.png").resolve()
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=520,1480", "--virtual-time-budget=4000", f"--screenshot={out}",
                    "http://127.0.0.1:8765/"], capture_output=True, timeout=60)
    print(scene, "ok" if out.exists() else "FAILED")
server.shutdown()
