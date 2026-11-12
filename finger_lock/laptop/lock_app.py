"""The finger lock, run from the laptop: the board's webcam -> the neural net -> the safe.

    python3 laptop/lock_app.py
    python3 laptop/lock_app.py --board 10.1.2.3 --tester "Sam"

1. Press the button on the safe. The window says "show number 1 of 3".
2. Hold up fingers in the red square and move your hand around a little. The first SETTLE_S
   seconds are skipped (the hand is still sliding in, fingers half-visible). Then the net
   averages how sure it is of each answer over the last WINDOW_S seconds of pictures. After at
   least MIN_S seconds of looking, as soon as the best answer's average reaches SURE, that
   number is accepted. If it never gets that sure, the lock still decides after MAX_S seconds,
   using the average of all the pictures: it never just sits there. (Seconds, not picture
   counts, so a faster or slower Wi-Fi doesn't change how long it takes.) A wrong guess only means a wrong code (the safe stays locked).
   The safe flashes a check mark, beeps, and adds a block (not the number) to the LED screen.
3. Take your hand out of the square (EMPTY pictures of an empty square in a row) before the next
   number, so 3-3-1 works. One stray "0" picture while your hand moves doesn't count.
4. After 3 numbers the safe checks the code: open, or X.

Who is using it: type a name in "Who's using it?" on the web page (or start with --tester).
It goes into test_log.csv and into the file name of every correction picture.

Green box: r fits it inside the red square (also done at the start), or click-drag a new box
with the mouse if r can't find the square. Either way it's saved for next time.

Keys: r fits the box to the red square again (it also does this when it starts; press r any
time the camera gets bumped). For class testing: 0-5 saves the current picture with the RIGHT
answer to laptop/data/ (do this when the net is wrong). q quits. Every attempt is logged to laptop/test_log.csv.
"""
import argparse
import csv
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "python"))
from fingernet import FingerNet, find_red_square, prepare  # noqa: E402

SETTLE_S = 1.0       # seconds skipped while the hand slides into the square
MIN_S = 3.0          # look at least this long before accepting a number
WINDOW_S = 2.0       # average the pictures from the last this-many seconds
SURE = 0.65          # accept once the winning answer's average is at least this sure
MAX_S = 7.0          # ... otherwise decide anyway after this long, using every picture
EMPTY = 3            # pictures of an empty square in a row = the hand is gone

parser = argparse.ArgumentParser()
parser.add_argument("--board", default="10.247.137.100")
parser.add_argument("--tester", default="Tasha", help="whose hand, if the web page doesn't say")
args = parser.parse_args()
URL = f"http://{args.board}:8000"

net = FingerNet()
net.load_state_dict(torch.load(HERE.parent / "python" / "finger_cnn.pt"))
net.eval()
MODEL_INFO = HERE.parent / "python" / "finger_cnn.json"
roi = json.loads(MODEL_INFO.read_text())["roi"]
LOG = HERE / "test_log.csv"


def ask(path="/state", method="GET"):
    req = urllib.request.Request(URL + path, method=method)
    with urllib.request.urlopen(req, timeout=3) as r:
        return r.read() if path.startswith("/frame") else json.load(r)


box_note, box_note_until = "", 0.0      # shown in the window after the box changes


def set_box(x, y, w, h, how):
    """Use (x, y, w, h) as the green box from now on, and save it for next time."""
    global box_note, box_note_until
    roi.update(x=int(x), y=int(y), w=int(w), h=int(h))
    info = json.loads(MODEL_INFO.read_text())
    info["roi"] = roi
    MODEL_INFO.write_text(json.dumps(info, indent=2))
    (HERE / "roi.json").write_text(json.dumps(roi))
    print(f"box {how}:", roi)
    box_note, box_note_until = f"box {how}", time.time() + 2


def refit_box(img, careful=False):
    """Fit the box inside the red square again (the camera got bumped). careful=True (at start-up)
    keeps the old box if the new one is a very different size: something was probably covering
    the square, like a hand."""
    global box_note, box_note_until
    found = find_red_square(img)
    if not found or found[2] < 150 or found[3] < 80:
        print("couldn't find the red square; is all of it in the picture? (or drag a box with the mouse)")
        box_note, box_note_until = "no red square found - drag a box", time.time() + 3
        return
    if careful and (abs(found[2] - roi["w"]) > 0.1 * roi["w"] or abs(found[3] - roi["h"]) > 0.1 * roi["h"]):
        print(f"start-up: the red square looks odd {found}, keeping the saved box {roi}. Press r to fit it anyway.")
        return
    set_box(*found, "fitted to the red square")


drag_start = drag_now = None


def on_mouse(event, x, y, flags, _):
    """Click-drag in the window to draw the green box by hand."""
    global drag_start, drag_now
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = drag_now = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drag_start:
        drag_now = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drag_start:
        w, h = abs(x - drag_start[0]), abs(y - drag_start[1])
        if w >= 150 and h >= 80:                     # ignore plain clicks and tiny boxes
            set_box(min(x, drag_start[0]), min(y, drag_start[1]), w, h, "drawn by hand")
        drag_start = drag_now = None


def who():
    """Whose hand: the name typed on the web page, or --tester."""
    return (latest["state"] or {}).get("tester") or args.tester


def log_attempt(digits, confidences, result):
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "tester", "numbers read", "confidence", "result"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), who(), digits,
                    " ".join(f"{c:.0%}" for c in confidences), result])


# ---------------------------------------------------------------- talking to the safe in the background
# Downloading pictures and asking the safe how it's doing both wait on Wi-Fi. They run in their own
# threads, so the main loop only ever works on the newest picture and never waits for the network.
latest = {"img": None, "count": 0, "state": None}
report = {"conf": 0.0, "looked": 0.0, "lead": 0, "phase": ""}   # sent to the web page: the running vote


def picture_thread():
    while True:
        try:
            img = cv2.imdecode(np.frombuffer(ask("/frame.jpg"), np.uint8), cv2.IMREAD_COLOR)
            latest["img"], latest["count"] = img, latest["count"] + 1
        except OSError:
            print("can't reach the safe's camera, retrying...")
            time.sleep(1)


def state_thread():
    while True:
        try:
            latest["state"] = ask(f"/state?conf={report['conf']:.3f}&votes={report['looked']:.1f}"
                                  f"&of={MAX_S}&sure={SURE}&lead={report['lead']}&phase={report['phase']}")
        except OSError:
            print("can't reach the safe, retrying...")
            time.sleep(1)
        time.sleep(0.1)


threading.Thread(target=picture_thread, daemon=True).start()
threading.Thread(target=state_thread, daemon=True).start()

entered, confidences = "", []
votes = []                               # (time, the net's answer scores) for each picture of this number
hand_since = None                        # when the hand appeared in the square
ready = True                             # True once the square has been empty: the next number can start
zero_run = 0                             # empty-square pictures in a row
message, message_until = "", 0.0
mode = "locked"
seen = 0                                 # the last picture we looked at
last_send = 0.0                          # when we last sent the safe a number

print("Press the button on the safe to start. Keys: r refit box, 0-5 save a corrected picture, q quits.")
while latest["img"] is None or latest["state"] is None:
    time.sleep(0.05)
refit_box(latest["img"], careful=True)   # line up with the red square every start
cv2.namedWindow("finger lock")
cv2.setMouseCallback("finger lock", on_mouse)
img, crop, guess = latest["img"], None, 0


def handle_key(key):
    """Returns False to quit. Checked every time we wait, so no key press gets lost."""
    if key == ord("q"):
        return False
    if key == ord("r"):
        refit_box(img)
    if ord("0") <= key <= ord("5") and crop is not None:   # the net was wrong: save the right answer
        name = "".join(c for c in who() if c.isalnum()) or "unknown"
        path = HERE / "data" / chr(key) / f"{time.time():.3f}_{name}.jpg"   # time first: still sorts by time
        cv2.imwrite(str(path), crop)
        print(f"saved a picture of {chr(key)} from {who()} for retraining (net said {guess})")
    return True


while True:
    if latest["count"] == seen:          # no new picture yet: don't vote twice on the same one
        if not handle_key(cv2.waitKey(10) & 0xFF):
            break
        continue
    seen, img, state = latest["count"], latest["img"], latest["state"]

    # The safe went back to locked/unlocked: the code was checked (or it timed out).
    if mode == "listening" and state["mode"] != "listening":
        if len(entered) == 3:
            result = "OPEN" if state["mode"] == "unlocked" else "WRONG CODE"
            log_attempt(entered, confidences, result)
            left = state.get("tries_left", 3)
            message = ("OPEN" if result == "OPEN"
                       else "3 WRONG IN A ROW - LOCKED OUT" if state.get("last") == "lockout"
                       else f"WRONG CODE - {left} {'try' if left == 1 else 'tries'} left")
            message_until = time.time() + 3
        entered, confidences = "", []
    if state["mode"] == "listening" and mode != "listening":
        entered, confidences, votes, ready, zero_run, hand_since = "", [], [], True, 0, None
    mode = state["mode"]

    # The safe is the boss of how many numbers are in. If it disagrees with us for more than a
    # second (a number got lost on Wi-Fi, or the button / "Start over" restarted the code), go with
    # the safe's count, so we never sit at "checking..." while the safe waits for another number.
    if mode == "listening" and time.time() - last_send > 1.0 and state["entered"] != len(entered):
        print(f"out of step: the safe has {state['entered']} numbers, we had {len(entered)}; going with the safe")
        keep = state["entered"]
        entered = (entered + "?" * keep)[:keep]          # "?" = a number the safe got but we lost track of
        confidences = (confidences + [0.0] * keep)[:keep]
        ready, votes, hand_since = False, [], None       # take your hand away, then the next number

    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    crop = img[y:y + h, x:x + w].copy()
    with torch.no_grad():
        probs = torch.softmax(net(prepare(crop)), dim=1)[0].numpy()
    guess, conf = int(probs.argmax()), float(probs.max())

    # ---------------------------------------------------------------- the lock policy
    average = None
    if mode == "listening" and len(entered) < 3:
        if guess == 0:
            zero_run += 1
            if zero_run >= EMPTY:                    # the hand is really gone: ready for the next number
                ready, votes, hand_since = True, [], None
        else:
            zero_run = 0
            now = time.time()
            if ready and hand_since is None:
                hand_since = now
            if ready and now - hand_since >= SETTLE_S:   # past the sliding-in part: collect votes
                votes.append((now, probs))
                looked = now - votes[0][0]
                average = np.mean([p for t, p in votes if now - t <= WINDOW_S], axis=0)
                pick = int(average[1:].argmax()) + 1     # best answer 1-5 (0 can't be part of a code)
                sure_enough = looked >= MIN_S and average[pick] >= SURE
                if not sure_enough and looked >= MAX_S:  # taking too long: decide anyway
                    average = np.mean([p for t, p in votes], axis=0)
                    pick = int(average[1:].argmax()) + 1
                if sure_enough or looked >= MAX_S:
                    try:
                        reply = ask(f"/digit?n={pick}&conf={average[pick]:.3f}", "POST")
                    except OSError:
                        print("couldn't send the number to the safe, trying again")
                        continue
                    last_send = time.time()
                    if reply.get("entered", 0) <= len(entered):  # the safe didn't take it
                        print(f"the safe didn't take {pick}: {reply.get('why', 'unknown reason')}")
                        ready, votes, average, hand_since = False, [], None, None
                        continue
                    entered += str(pick)
                    confidences.append(float(average[pick]))
                    how = "sure enough" if sure_enough else "out of time, best guess"
                    print(f"accepted {pick} ({average[pick]:.0%} sure after {looked:.1f} s, "
                          f"{len(votes)} pictures, {how})")
                    ready, votes, average, hand_since = False, [], None, None
    report["conf"] = float(average[1:].max()) if average is not None else 0.0
    report["lead"] = int(average[1:].argmax()) + 1 if average is not None else 0
    report["phase"] = ("" if mode != "listening" or len(entered) >= 3
                       else "away" if not ready                  # take your hand out of the square
                       else "reading" if votes
                       else "settle" if hand_since is not None
                       else "show")
    report["looked"] = time.time() - votes[0][0] if votes else 0.0

    # ---------------------------------------------------------------- the window
    view = img.copy()
    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(view, f"this picture: {guess} ({conf:.0%})", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if average is not None:                          # the running vote for this number
        lead = int(average[1:].argmax()) + 1
        cv2.putText(view, f"vote: {lead} ({average[lead]:.0%})", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if time.time() < message_until:
        status = message
    elif mode == "listening":
        if len(entered) >= 3:
            status = "checking..."
        elif not ready:
            status = "take your hand away"
        else:
            status = f"show number {len(entered) + 1} of 3 - move your hand a little"
    elif mode == "unlocked":
        status = "OPEN - press button to lock"
    elif state.get("lockout_seconds"):
        status = f"locked out: try again in {state['lockout_seconds']} s"
    else:
        status = "press the button on the safe"
    cv2.putText(view, status, (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(view, "  ".join(["*"] * len(entered) + ["_"] * (3 - len(entered))),
                (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    if votes:                                        # progress: green = normal vote, orange = running out of time
        looked = report["looked"]
        cv2.rectangle(view, (10, 50), (10 + int(200 * min(looked, MAX_S) / MAX_S), 62),
                      (0, 255, 0) if looked <= MIN_S + 1 else (0, 165, 255), -1)
    if drag_start:                                   # the box being dragged right now
        cv2.rectangle(view, drag_start, drag_now, (255, 0, 255), 2)
    if time.time() < box_note_until:
        cv2.putText(view, box_note, (330, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.imshow("finger lock", view)

    if not handle_key(cv2.waitKey(1) & 0xFF):
        break

cv2.destroyAllWindows()
