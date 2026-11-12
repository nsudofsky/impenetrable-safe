"""Finger lock, Linux side: the safe's brain.

Press the button -> the safe listens for ENTRY_SECONDS. Whoever reads the fingers (for now the
laptop, later the camera + neural net on this board) sends one number at a time; after three we
unlock only if they match the passcode. Press the button again once the lid is shut to lock.

The LED screen never shows the numbers (it's a safe): a block per number entered, and a check
mark flash each time a number is read. Only the end says whether the whole code was right, so
nobody can find the code one digit at a time. MAX_WRONG wrong codes in a row lock the safe out
for LOCKOUT_SECONDS (button and web Start do nothing), so nobody can just try all 125 codes.

Web page and API (port 8000):
    GET  /               -> the web page (web.html): status, camera, start/lock, change the code
    GET  /state          -> {"mode": "locked" | "listening" | "unlocked", "entered": 0-3,
                             "last": "open" | "wrong" | "lockout" | null, "tries_left": 0-3,
                             "lockout_seconds": 0-30, "reading": {"conf", "votes", "of"} or null,
                             "confidences": [...]}
                            The laptop adds ?conf=0.72&votes=1.2&of=5&sure=0.65&lead=3: how sure its
                            running vote is, seconds looked of the most it will look, the bar to
                            accept, the number it's leaning toward (the web page can hide it), and
                            the step: show / settle / reading / away (take your hand out).
    POST /digit?n=3&conf=0.81 -> {"ok": true/false/null, ...}   (ok is null until 3 numbers are in)
    POST /start          -> start listening (same as the arcade button)
    POST /lock           -> lock
    POST /passcode?code=213 -> set a new code; only while the safe is open
    POST /tester?name=Sam -> who is using the lock now (the laptop logs attempts and labels
                             correction pictures with it)
    GET  /frame.jpg      -> the newest webcam picture
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from arduino.app_utils import *
from fingernet_np import FingerNetNumpy, find_red_square

PASSCODE_FILE = Path(__file__).with_name("passcode.txt")   # survives restarts; made by the web page
DEFAULT_CODE = "324"    # used until a new code is set
ENTRY_SECONDS = 60      # how long the safe listens after the button
MAX_WRONG = 3           # wrong codes in a row before a lockout
LOCKOUT_SECONDS = 10    # how long the lockout lasts (short, for demos)
PORT = 8000
CAMERA = 0              # /dev/video0 = the USB webcam on the hub

# The UNO Q reads the fingers itself with the trained net (numpy). Same policy as laptop/lock_app.py:
SETTLE_S = 1.0          # seconds skipped while the hand slides into the square
MIN_S = 3.0             # look at least this long before accepting a number
WINDOW_S = 2.0          # average the pictures from the last this-many seconds
SURE = 0.65             # accept once the winning answer's average is at least this sure
MAX_S = 7.0             # ... otherwise decide anyway after this long, using every picture
EMPTY = 3               # empty-square pictures in a row = the hand is gone

LOCK_ICON, OPEN_ICON, CHECK_ICON, X_ICON = 0, 1, 2, 3

mode = "locked"
entered = ""
listening_since = 0.0
last_result = None      # "open" / "wrong" after a code is checked, for the web page
confidences = []        # how sure the net was of each accepted number (not the numbers)
reading = None          # the laptop's running vote: {"conf", "votes", "of", "at"}
wrong_in_a_row = 0
lockout_until = 0.0
tester = ""             # who is using the lock, set on the web page ("" = the laptop's --tester)


def passcode():
    try:
        code = PASSCODE_FILE.read_text().strip()
        if len(code) == 3 and all(c in "12345" for c in code):
            return code
    except OSError:
        pass
    return DEFAULT_CODE


def valid_code(code):
    return len(code) == 3 and all(c in "12345" for c in code)


def lockout_left():
    """Seconds of lockout left (0 = not locked out)."""
    return max(0, int(lockout_until - time.time() + 0.999))


latest = {"img": None, "n": 0}          # newest camera picture and its number
jpeg_cache = {}                          # (picture number, small?) -> JPEG bytes
jpeg_lock = threading.Lock()


def camera_loop():
    """Keep grabbing pictures so /frame.jpg is always the newest one. Pictures are only turned
    into JPEGs when someone asks (compressing all 30 a second kept the board's CPU busy)."""
    while True:
        cam = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("camera open:", cam.isOpened(), flush=True)
        while cam.isOpened():
            ok, img = cam.read()
            if not ok:
                break
            latest["img"], latest["n"] = img, latest["n"] + 1
        cam.release()
        print("camera lost, retrying", flush=True)
        time.sleep(2)


def newest_jpeg(small=False):
    """The newest picture as a JPEG: full size for the neural net, small for the web page."""
    img, n = latest["img"], latest["n"]
    if img is None:
        return b""
    with jpeg_lock:
        key = (n, small)
        if key not in jpeg_cache:
            if small:
                img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
            if len(jpeg_cache) > 4:                             # only the newest pictures matter
                jpeg_cache.clear()
            jpeg_cache[key] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70 if small else 90])[1].tobytes()
        return jpeg_cache[key]


def bridge(*args):
    Bridge.call(*args, timeout=1)


def set_mode(new_mode):
    global mode, entered, listening_since, confidences
    mode, entered, listening_since = new_mode, "", time.time()
    if mode == "listening":
        confidences = []
    bridge("set_locked", mode != "unlocked")
    bridge("set_light", mode == "listening")
    if mode == "listening":
        bridge("progress", 0)                                 # "_ _ _"
    else:
        bridge("show", OPEN_ICON if mode == "unlocked" else LOCK_ICON)
    print("safe is now", mode, flush=True)


def finish(ok):
    global last_result, wrong_in_a_row, lockout_until
    wrong_in_a_row = 0 if ok else wrong_in_a_row + 1
    locked_out = wrong_in_a_row >= MAX_WRONG
    print("code entered:", "correct" if ok else f"wrong ({wrong_in_a_row} in a row)", flush=True)
    last_result = "open" if ok else "lockout" if locked_out else "wrong"
    if locked_out:
        lockout_until = time.time() + 2.5 + LOCKOUT_SECONDS
    time.sleep(1.0)                                           # let the last check mark flash finish
    bridge("show", CHECK_ICON if ok else X_ICON)
    bridge("beep", 2500 if ok else 300, 400 if ok else 700)  # high = open, low buzz = wrong
    time.sleep(1.5)
    set_mode("unlocked" if ok else "locked")
    if locked_out:
        print(f"{MAX_WRONG} wrong codes in a row: locked out for {LOCKOUT_SECONDS} s", flush=True)
        for left in range(LOCKOUT_SECONDS, 0, -1):            # LED screen counts down 10, 9, 8...
            bridge("countdown", left)
            time.sleep(1)
        wrong_in_a_row = 0
        if mode == "locked":
            bridge("show", LOCK_ICON)
        print("lockout over", flush=True)


def on_button(_):
    # Runs in a thread: Bridge.call can't be used directly inside a Bridge callback.
    if lockout_left():
        threading.Thread(target=bridge, args=("beep", 300, 300)).start()   # locked out: just a low buzz
        return
    threading.Thread(target=set_mode, args=("locked" if mode == "unlocked" else "listening",)).start()


Bridge.provide("button", on_button)


class Handler(BaseHTTPRequestHandler):
    def reply(self, data, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(200 if body else 503)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def state(self, **extra):
        live = reading if reading and mode == "listening" and time.time() - reading["at"] < 2 else None
        return {"mode": mode, "entered": len(entered), "last": last_result, "confidences": confidences,
                "tries_left": max(0, MAX_WRONG - wrong_in_a_row), "lockout_seconds": lockout_left(),
                "lockout_length": LOCKOUT_SECONDS, "tester": tester,
                "reading": live and {k: live[k] for k in ("conf", "votes", "of", "sure", "lead", "phase")}, **extra}

    def do_GET(self):
        global reading
        path = urlparse(self.path).path
        if path == "/frame.jpg":
            self.reply(newest_jpeg(small="small" in self.path), "image/jpeg")
        elif path == "/":
            self.reply(Path(__file__).with_name("web.html").read_bytes(), "text/html; charset=utf-8")
        else:                                                  # /state
            q = parse_qs(urlparse(self.path).query)
            if "conf" in q:                                    # the laptop reporting its running vote
                try:
                    reading = {"conf": float(q["conf"][0]), "votes": float(q["votes"][0]),
                               "of": float(q["of"][0]), "sure": float(q.get("sure", ["0.65"])[0]),
                               "lead": int(q.get("lead", ["0"])[0]), "phase": q.get("phase", [""])[0],
                               "at": time.time()}
                except (KeyError, ValueError):
                    pass
            if mode == "listening" and time.time() - listening_since > ENTRY_SECONDS:
                set_mode("locked")                             # nobody entered a code in time
            self.reply(self.state())

    def do_POST(self):
        global entered, last_result, tester
        url = urlparse(self.path)
        query = parse_qs(url.query)

        if url.path == "/start":
            if lockout_left():
                self.reply(self.state(ok=False, why=f"locked out: try again in {lockout_left()} s"))
                return
            if mode == "unlocked":
                self.reply(self.state(ok=False, why="the safe is already open"))
                return
            last_result = None
            set_mode("listening")
            self.reply(self.state(ok=True))
        elif url.path == "/lock":
            set_mode("locked")
            self.reply(self.state(ok=True))
        elif url.path == "/tester":
            name = " ".join(query.get("name", [""])[0].split())[:30]
            tester = "".join(c for c in name if c.isalnum() or c in " -'")
            print("now testing:", tester or "(laptop default)", flush=True)
            self.reply(self.state(ok=True))
        elif url.path == "/passcode":
            code = query.get("code", [""])[0]
            if mode != "unlocked":
                self.reply(self.state(ok=False, why="open the safe first, then you can change the code"))
            elif not valid_code(code):
                self.reply(self.state(ok=False, why="the code is 3 numbers, each 1-5"))
            else:
                PASSCODE_FILE.write_text(code)
                print("new passcode set", flush=True)          # never print the code itself
                self.reply(self.state(ok=True))
        else:                                                  # /digit
            try:
                conf = float(query.get("conf", ["0"])[0])
            except ValueError:
                conf = 0.0
            ok, why = enter_digit(query.get("n", [""])[0], conf)
            self.reply(self.state(ok=ok, **({"why": why} if why else {})))

    def log_message(self, *args):                              # keep the app log quiet
        pass


digit_lock = threading.Lock()


def enter_digit(n, conf):
    """One number read (by the board's own net, or sent by the laptop). Returns (ok, why):
    ok is None until 3 numbers are in, then True/False; why says why a number was refused."""
    global entered
    with digit_lock:
        if mode != "listening" or len(entered) >= 3:
            return None, "press the button first"
        if len(n) != 1 or n not in "12345":
            return None, "numbers are 1-5"
        entered += n
        confidences.append(round(conf, 3))
        bridge("progress", len(entered))
        bridge("flash", 2)                                     # check mark + beep: number read
        if len(entered) < 3:
            return None, ""
        ok = entered == passcode()
        threading.Thread(target=finish, args=(ok,)).start()
        return ok, ""


def reader_loop():
    """The UNO Q's own finger reader: camera -> numpy net -> vote -> enter_digit."""
    global reading
    net = FingerNetNumpy()
    info_file = Path(__file__).with_name("finger_cnn.json")
    roi = json.loads(info_file.read_text())["roi"]
    print("finger reader ready (numpy net on the UNO Q)", flush=True)
    seen, session = 0, None
    votes, ready, zero_run, hand_since, count = [], True, 0, None, 0
    while True:
        if mode != "listening" or latest["img"] is None or latest["n"] == seen or len(entered) >= 3:
            time.sleep(0.02)
            continue
        if session != listening_since:                         # a new code entry: start fresh
            session = listening_since
            votes, ready, zero_run, hand_since, count = [], True, 0, None, 0
            found = find_red_square(latest["img"])             # camera bumped? re-fit the box
            if found and abs(found[2] - roi["w"]) < 0.1 * roi["w"] and abs(found[3] - roi["h"]) < 0.1 * roi["h"]:
                roi = {"x": found[0], "y": found[1], "w": found[2], "h": found[3]}
        if len(entered) != count:                              # a number went in: hand away first
            count, votes, ready, hand_since = len(entered), [], False, None
        seen, img = latest["n"], latest["img"]
        started = time.time()
        probs = net.probs(img[roi["y"]:roi["y"] + roi["h"], roi["x"]:roi["x"] + roi["w"]])
        think_ms = (time.time() - started) * 1000
        guess, now = int(probs.argmax()), time.time()

        average = None
        if guess == 0:
            zero_run += 1
            if zero_run >= EMPTY:                              # the hand is really gone
                ready, votes, hand_since = True, [], None
        else:
            zero_run = 0
            if ready and hand_since is None:
                hand_since = now
            if ready and now - hand_since >= SETTLE_S:         # past the sliding-in part: vote
                votes.append((now, probs))
                looked = now - votes[0][0]
                average = np.mean([p for t, p in votes if now - t <= WINDOW_S], axis=0)
                pick = int(average[1:].argmax()) + 1           # 1-5 (0 can't be part of a code)
                sure_enough = looked >= MIN_S and average[pick] >= SURE
                if not sure_enough and looked >= MAX_S:        # taking too long: best guess
                    average = np.mean([p for t, p in votes], axis=0)
                    pick = int(average[1:].argmax()) + 1
                if sure_enough or looked >= MAX_S:
                    print(f"board read a number ({average[pick]:.0%} sure after {looked:.1f} s, "
                          f"{think_ms:.0f} ms per picture)", flush=True)
                    enter_digit(str(pick), float(average[pick]))
                    average = None

        phase = ("away" if not ready else "reading" if votes else "settle" if hand_since else "show")
        reading = {"conf": float(average[1:].max()) if average is not None else 0.0,
                   "votes": now - votes[0][0] if votes else 0.0, "of": MAX_S, "sure": SURE,
                   "lead": int(average[1:].argmax()) + 1 if average is not None else 0,
                   "phase": phase, "at": time.time()}


threading.Thread(target=reader_loop, daemon=True).start()
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever, daemon=True).start()
App.run()
