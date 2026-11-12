"""Collect training pictures of fingers for the lock's neural net.

The webcam is on the UNO Q's hub, so the pictures come from the board over WiFi (the same camera
the lock will use). Live view with a box (the area the net will look at). Press r to fit the box inside the red square
automatically, or click-drag your own; it's saved to roi.json so the lock uses the same area.

    python3 laptop/collect.py                       # webcam on the board (finger_lock app running)
    python3 laptop/collect.py --board 10.1.2.3      # if the board's IP changed
    python3 laptop/collect.py --camera 1            # a webcam plugged into the laptop instead

Keys:  0-5  save a picture of that many fingers (0 = no hand); hold the key to save a burst
       r    find the red square          u  undo the last picture        q  quit
Pictures go to laptop/data/<fingers>/.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "python"))
from fingernet import find_red_square  # noqa: E402
DATA = HERE / "data"
ROI_FILE = HERE / "roi.json"

parser = argparse.ArgumentParser()
parser.add_argument("--board", default="10.247.137.100")
parser.add_argument("--camera", type=int, help="use a webcam plugged into the laptop instead of the board's")
args = parser.parse_args()

if args.camera is not None:
    cam = cv2.VideoCapture(args.camera)

    def grab():
        ok, img = cam.read()
        return img if ok else None
else:
    url = f"http://{args.board}:8000/frame.jpg"

    def grab():
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                return cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            return None

img = grab()
if img is None:
    raise SystemExit("no picture: is the finger_lock app running on the board (or the --camera number right)?")
h, w = img.shape[:2]

# The box (x, y, w, h) in camera pixels.
if ROI_FILE.exists():
    roi = json.loads(ROI_FILE.read_text())
else:
    roi = {"x": w // 4, "y": h // 4, "w": w // 2, "h": h // 2}


def save_roi(x, y, bw, bh):
    if bw < 150 or bh < 80:
        print("that box is too small, keeping the old one")
        return
    if any((DATA / str(n)).glob("*.jpg") for n in range(6)):
        print("WARNING: you already have pictures; ones taken with a different box won't match. "
              "Best to delete laptop/data and start over if the camera moved.")
    roi.update(x=int(x), y=int(y), w=int(bw), h=int(bh))
    ROI_FILE.write_text(json.dumps(roi))
    print("box saved:", roi)


drag_start = drag_now = None


def on_mouse(event, x, y, flags, _):
    global drag_start, drag_now
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = drag_now = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drag_start:
        drag_now = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drag_start:
        bw, bh = abs(x - drag_start[0]), abs(y - drag_start[1])
        if bw > 30 and bh > 30:                       # ignore plain clicks (like clicking to focus the window)
            save_roi(min(x, drag_start[0]), min(y, drag_start[1]), bw, bh)
        drag_start = drag_now = None


for n in range(6):
    (DATA / str(n)).mkdir(parents=True, exist_ok=True)
saved = []                                            # for undo

cv2.namedWindow("collect")
cv2.setMouseCallback("collect", on_mouse)
print(f"picture {w}x{h}. Press 0-5 to save, u to undo, q to quit.")

while True:
    img = grab()
    if img is None:
        print("lost the picture, retrying...")
        time.sleep(1)
        continue
    x, y, bw, bh = roi["x"], roi["y"], roi["w"], roi["h"]
    crop = img[y:y + bh, x:x + bw]

    view = img.copy()
    cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    if drag_start:                                    # the box you're dragging right now
        cv2.rectangle(view, drag_start, drag_now, (255, 0, 255), 2)
    counts = "  ".join(f"{n}:{len(list((DATA / str(n)).glob('*.jpg')))}" for n in range(6))
    cv2.putText(view, counts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("collect", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if ord("0") <= key <= ord("5") and crop.size:
        path = DATA / chr(key) / f"{time.time():.3f}.jpg"
        cv2.imwrite(str(path), crop)
        saved.append(path)
    if key == ord("r"):
        found = find_red_square(img)
        if found:
            save_roi(*found)
        else:
            print("no red square found; is all of it in the picture?")
    if key == ord("u") and saved:
        saved.pop().unlink(missing_ok=True)

cv2.destroyAllWindows()
